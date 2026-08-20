classdef Predictor < TemporalNeuralNet
    %% PREDICTOR: forecasting subclass of TemporalNeuralNet
    %
    % Implements the sequence-forecasting specifics: autoregressive multi-step forecast(),
    % multi-step/teacher-forced train(), MSE loss/backprop, and MAE evaluation.
    %
    % See TemporalNeuralNet.m for the shared CNN->RNN->FC backbone (forward()), Adam
    % optimizer, and memory/gradient reset helpers.
    %
    % forward() vs. forecast(): forward() (inherited) is a single pass with true historical
    % values as input, the last output step is the prediction. forecast() is the
    % autoregressive, multi-step rollout: it feeds the last steps' own prediction back in
    % as input for the next step. train() uses forward() directly for its per-window
    % predictions, but its outer loop performs numAutoregressiveSteps of these, each one
    % with the context window rolled forward using the network's own prediction.
    %
    % DATA FORMAT: trainingData/validationData/testData are N x 1 cell arrays; each element
    % is one trial sequence, [T x numChannels].
    properties
        forecastLength    % target evaluation horizon, used for the constructor's initial
                          % baseline and for per-epoch validation reporting in train(), not
                          % for training-window construction, that's numAutoregressiveSteps.
        trainingLength    % Context window length (number of steps fed in before the model
                          % has to predict anything). Used by: the constructor's initial
                          % baseline evaluate() call, segmentSequences() (via train()), and
                          % evaluate()'s own internal forecast() call.
        numAutoregressiveSteps   % Default number of autoregressive steps train() performs
                          % per window when the caller doesn't override it via the
                          % 'numAutoregressiveSteps' name-value argument. 1 = pure teacher
                          % forcing.
    end

    methods
        %% Predictor constructor
        % Inputs:
        %   validationData - N x 1 cell array, used to compute the initial baseline MAE
        %                    stored as the first row of learningHistory.
        %   'forecastLength'          - target horizon (default 12)
        %   'trainingLength'          - context window length (default 24)
        %   'numAutoregressiveSteps'  - default steps for train() (default 1)
        %   (anything else is forwarded to TemporalNeuralNet's constructor)
        %
        % Output: obj, ready to train() or forecast()
        function obj = Predictor(validationData, varargin)
            p = inputParser;
            p.KeepUnmatched = true;   % everything else forwards to the parent
            addParameter(p,'forecastLength', 12, @(x) isnumeric(x)&&isscalar(x));
            addParameter(p,'trainingLength', 24, @(x) isnumeric(x)&&isscalar(x));
            addParameter(p,'numAutoregressiveSteps', 1, @(x) isnumeric(x)&&isscalar(x));
            parse(p, varargin{:});

            % Forward CNN/RNN/FC/tPool/numChannels/eta/... to the shared parent constructor.
            names = fieldnames(p.Unmatched);
            vals  = struct2cell(p.Unmatched);
			
            unmatchedArgs = [names vals]';
            unmatchedArgs = unmatchedArgs(:)';
            obj@TemporalNeuralNet(unmatchedArgs{:});

            obj.forecastLength         = p.Results.forecastLength;
            obj.trainingLength         = p.Results.trainingLength;
            obj.numAutoregressiveSteps = p.Results.numAutoregressiveSteps;
            obj.outputWidth            = obj.numChannels;

            initialMSE = obj.evaluate(validationData, obj.forecastLength, obj.trainingLength);
            obj.learningHistory = [0, obj.forecastLength, 0, initialMSE, NaN, NaN, NaN];

        end

        %% outputActivation: no activation on the raw FC output
        % See TemporalNeuralNet.forward for where this gets called.
        function a = outputActivation(~, z)
            a = z;
        end

        %% forecast: Autoregressive multi-step forecast (single sequence or batch)
        % Feeds each step's own prediction back in as input for the next step. This is
        % inference rollout, distinct from train()'s rollout.
        %
        % Inputs:
        %   inputData       - Either a [T x numChannels] raw sequence, OR an N x 1
        %                     cell array of sequences (each element a trial).
        %   'isTraining'    - (default false) passed through to each internal forward()
        %                     call, true enables activation caching.
        %   'forecastLength' - (default 1) how many steps ahead to forecast.
        %   'segmentLength' - (default 24) how many raw steps of context to use, taken
        %                     from immediately before the forecast region.
        %
        % Output:
        %   networkOutput   - same "shape" as inputData: a raw array if given an array,
        %                     or an N x 1 cell of arrays if given a cell array. Each
        %                     output matches its input's size, with only the last
        %                     forecastLength rows replaced by the actual forecast and
        %                     earlier rows are an unchanged copy of the input.
        function networkOutput = forecast(obj, inputData, varargin)
            p = inputParser;
            addParameter(p,'isTraining',     false,@(x) islogical(x));
            addParameter(p,'forecastLength', 1,    @(x) isnumeric(x)&&isscalar(x));
            addParameter(p,'segmentLength',  24,    @(x) isnumeric(x)&&isscalar(x));
            parse(p, varargin{:});
            isTraining    = p.Results.isTraining;
            forecastL     = p.Results.forecastLength;
            segmentLength = p.Results.segmentLength;

            wasCell = iscell(inputData);
            if wasCell
                sequences = inputData(:,1);
            else
                sequences = {inputData};
            end

            outputs = cell(numel(sequences),1);
            for s = 1:numel(sequences)
                outputs{s} = obj.forecastSequence(sequences{s}, isTraining, forecastL, segmentLength);
            end

            if wasCell
                networkOutput = outputs;
            else
                networkOutput = outputs{1};
            end
        end

        %% forecastSequence: autoregressive rollout for a single raw sequence
        % Internal helper for forecast()
        %
        % Inputs:
        %   inputSequence - [T x numChannels] raw sequence
        %   isTraining    - passed through to each internal forward() call
        %   forecastL     - number of autoregressive steps to roll out
        %   segmentL      - context window length taken immediately before the forecast
        %                   region
        %
        % Output:
        %   networkOutput - same size as inputSequence, with the last forecastL rows 
        %                   replaced by the rollout
        function networkOutput = forecastSequence(obj, inputSequence, isTraining, forecastL, segmentL)
					 
            T = size(inputSequence,1);

            if isnan(segmentL)
                segmentL = T - forecastL;
            end

            startIdx = T - forecastL - segmentL + 1;
            endIdx   = T - forecastL;
            if startIdx < 1
                error('forward:insufficientLength', ...
                    ['inputSequence (length %d) is too short for ' ...
                     'segmLength (%d) and forecastLength (%d).'], ...
                     T, segmentL, forecastL);
            end

            networkOutput = inputSequence;
            window        = inputSequence(startIdx:endIdx, :);

            for k = 1:forecastL
                obj.resetMemory();
                stepOutput = obj.forward(window, 'isTraining',isTraining);
                prediction = stepOutput(end, :);

                window = [window(2:end, :); prediction];
                networkOutput(T-forecastL+k, :) = prediction;
            end
        end

        %% train: Optimize network via teacher-forced next-step prediction
        % For each training segment, performs numAutoregressiveSteps forward/backward
        % passes: each one predicts the raw value right after the current context window,
        % then the context window is rolled forward by one step using the network's own
        % prediction as the newest entry, and the process repeats. Gradients from all
        % numAutoregressiveSteps passes are summed before a single Adam update.
        %
        % Inputs:
        %   trainingData, validationData - N x 1 cell arrays, one row per independent
        %                 trial, [T x numChannels] each
        %   epochs      - number of full passes over trainingData
        %   batchSize   - training segments per Adam update
        %   'numAutoregressiveSteps' - (name-value pair, default
        %               obj.numAutoregressiveSteps) autoregressive steps per segment
        %
        % Output: none (mutates obj: weights, learningHistory, totalLossHistory,
        %         trainingMetricHistory; also saves a Predictor_trained_<timestamp>.mat
        %         checkpoint after every epoch).
        function train(obj, trainingData, validationData, epochs, batchSize, varargin)

            p = inputParser;
            addParameter(p,'numAutoregressiveSteps', obj.numAutoregressiveSteps, @(x) isnumeric(x)&&isscalar(x));
            parse(p, varargin{:});

            numAutoregSteps = p.Results.numAutoregressiveSteps;
            segmInfos = obj.segmentSequences(trainingData, obj.trainingLength, numAutoregSteps);

            for epochIdx = 1:epochs
                tic
                randIdxList = randperm(size(segmInfos,1));

                totalAE_epoch      = 0;
                totalSE_epoch      = 0;
                totalSamples_epoch = 0;

                for b = 1:ceil(size(segmInfos,1)/batchSize)
                    obj.resetGrads();

                    totalAE_batch      = 0;
                    totalSE_batch      = 0;
                    totalSamples_batch = 0;

                    rows      = randIdxList((b-1)*batchSize+1:min(b*batchSize,numel(randIdxList)));
                    trialIdxs = segmInfos(rows, 1);
                    startIdxs = segmInfos(rows, 2);
                    endIdxs   = segmInfos(rows, 3);

                    cnnUpdateSize = obj.idxMap.cnnStrt(end)-1;
                    fcUpdateSize  = obj.idxMap.fcStrt(end)-1;

                    cnnBatchUpdate = zeros(cnnUpdateSize,1);
                    fcBatchUpdate  = zeros(fcUpdateSize, 1);
                    rnnBatchUpdate = zeros(obj.idxMap.rnn{end}(end,2),1);

                    batchData = cell(numel(rows),1);

                    for i = 1:numel(rows)
                        trialIdx     = trialIdxs(i);
                        startIdx     = startIdxs(i);
                        endIdx       = endIdxs(i);
                        batchData{i} = trainingData{trialIdx,1}(startIdx:endIdx+numAutoregSteps,:);
                    end

                    % parfor
                    parfor segmIdx = 1:size(trialIdxs,1)
                   
                        trainSegment   = batchData{segmIdx};
                        contextSegment = trainSegment(1:end-numAutoregSteps, :);

                        for autoregStep = 1:numAutoregSteps
                            obj.resetMemory();

                            % build target sequence
                            targetSequence = trainSegment(autoregStep:end-numAutoregSteps+autoregStep,:);

                            output = obj.forward(contextSegment, 'isTraining', true);

                            [cnnSeqUpdate, rnnSeqUpdate, fcSeqUpdate, totalAE, totalSE, totalSamples] = obj.backwardPass(output, targetSequence);

                            contextSegment = [contextSegment(2:end,:); output(end,:)];
                         
                            % accumulate things
                            cnnBatchUpdate = cnnBatchUpdate + cnnSeqUpdate;
                            rnnBatchUpdate = rnnBatchUpdate + rnnSeqUpdate;
                            fcBatchUpdate  = fcBatchUpdate  + fcSeqUpdate;

                            totalAE_batch      = totalAE_batch      + totalAE;
                            totalSE_batch      = totalSE_batch      + totalSE;
                            totalSamples_batch = totalSamples_batch + totalSamples;
                        end
                    end

                    convLayerIdx = 0;

                    for i = 1:numel(obj.cnnModule)
                        if isa(obj.cnnModule{i}, "ConvolutionalLayer")
                            convLayerIdx = convLayerIdx + 1;
                            strtIdx      = obj.idxMap.cnnStrt(convLayerIdx);
                            endIdx       = obj.idxMap.cnnEnd(convLayerIdx);
                            numBiases    = numel(obj.cnnModule{i}.biases);
                            obj.cnnModule{i}.dW = reshape(cnnBatchUpdate(strtIdx:endIdx-numBiases),  size(obj.cnnModule{i}.dW));
                            obj.cnnModule{i}.db = reshape(cnnBatchUpdate(endIdx-numBiases+1:endIdx), size(obj.cnnModule{i}.db));
                        end
                    end

                    for i = 1:numel(obj.rnnModule)
                        for j = 1:numel(obj.rnnModule{i}.weights)
                            strtIdx   = obj.idxMap.rnn{i}(j,1);
                            endIdx    = obj.idxMap.rnn{i}(j,2);
                            numBiases = numel(obj.rnnModule{i}.biases{j});
                            obj.rnnModule{i}.dW{j} = reshape(rnnBatchUpdate(strtIdx:endIdx-numBiases),  size(obj.rnnModule{i}.dW{j}));
                            obj.rnnModule{i}.db{j} = reshape(rnnBatchUpdate(endIdx-numBiases+1:endIdx), size(obj.rnnModule{i}.db{j}));
                        end
                    end

                    for i = 1:numel(obj.fcModule.sizes)-1
                        strtIdx   = obj.idxMap.fcStrt(i);
                        endIdx    = obj.idxMap.fcEnd(i);
                        numBiases = numel(obj.fcModule.biases{i});
                        obj.fcModule.dW{i} = reshape(fcBatchUpdate(strtIdx:endIdx-numBiases),  size(obj.fcModule.dW{i}));
                        obj.fcModule.db{i} = reshape(fcBatchUpdate(endIdx-numBiases+1:endIdx), size(obj.fcModule.db{i}));
                    end

                    adamOptimizer(obj, totalSamples_batch);
                    obj.t = obj.t + 1;

                    obj.totalLossHistory      = [obj.totalLossHistory;      totalSE_batch/totalSamples_batch];
                    obj.trainingMetricHistory = [obj.trainingMetricHistory; totalAE_batch/totalSamples_batch];

                    % Plot progress
                    subplot(2,1,1); cla;
                    plot(obj.totalLossHistory);
                    grid on;
                    title('Training loss (MSE)');
                    xlabel('Batch'); ylabel('MSE');

                    subplot(2,1,2); cla;
                    plot(obj.trainingMetricHistory);
                    grid on;
                    title('Secondary metric (MAE)');
                    xlabel('Batch'); ylabel('MAE');

                    drawnow

                    totalAE_epoch      = totalAE_epoch      + totalAE_batch;
                    totalSE_epoch      = totalSE_epoch      + totalSE_batch;
                    totalSamples_epoch = totalSamples_epoch + totalSamples_batch;
                end

                elapsedTime = toc;
                fprintf('Epoch %d completed in %.0f s.\n', size(obj.learningHistory,1), elapsedTime);

                % Update learningHistory log
                trainingMAE   = totalAE_epoch/totalSamples_epoch;
                residual      = totalSE_epoch/totalSamples_epoch;

                validationMAE = obj.evaluate(validationData, obj.forecastLength, obj.trainingLength);
                obj.learningHistory(end+1,:) = [batchSize, numAutoregSteps, obj.eta, validationMAE, trainingMAE, residual, elapsedTime];
                fprintf('Validation MAE: %.4f\n', validationMAE);
                obj.eta = obj.learningRateDecay*obj.eta;

                timestamp = datetime('now', 'Format','yyyy-MM-dd_HH-mm-SS');
                filename  = sprintf('Predictor_trained_%s.mat', timestamp);
                save(filename, 'obj');
            end
        end

        %% evaluate: Compute average forecasting MAE over a dataset
        % For each trial, autoregressively forecasts the last forecastLength steps via
        % forecast() and compares against the trial's own true trailing values.
        %
        % Inputs:
        %   data           - N x 1 cell array, one row per trial, [T x numChannels] each
        %   forecastLength - number of trailing steps to forecast and score
        %
        % Output:
        %   MAE - mean absolute error, averaged over every forecasted step, normalized by
        %         numChannels.
        function MAE = evaluate(obj, data, forecastLength)
            n = size(data, 1);
            totalSE    = 0;
            totalSteps = 0;
 
            for i = 1:n
                input = data{i,1};
 
                if size(input,1) < forecastLength + obj.trainingLength
                    continue;
                end

                obj.resetMemory();

                output = obj.forecast(input, forecastLength=forecastLength, segmentLength=obj.trainingLength);

                trueVals = input(end-forecastLength+1:end, :);
                predVals = output(end-forecastLength+1:end, :);
 
                err = trueVals - predVals;
                totalSE    = totalSE + sum(abs(err(:)))/obj.numChannels;
                totalSteps = totalSteps + forecastLength;
            end
 
            if totalSteps == 0
                warning('No evaluable frames found.');
                MAE = NaN;
            else
                MAE = totalSE/totalSteps;
            end
        end
 
        %% backwardPass: runs backpropagation through all the layers of the network
        % Backprops one forward()-with-caching pass's worth of output against targetSequence
        % (the raw sequence output was predicting from), through the whole network, accumulating
        % flat idxMap-packed gradient vectors. Called once per autoregressive step inside train().
        %
        % Inputs:
        %   output         - forward()'s teacher-forced output, [numSteps x numChannels]
        %   targetSequence - the full raw segment (context + reserved trailing step(s)) that
        %                    output(j,:) was predicting from
        %
        % Outputs:
        %   cnnSeqUpdate, rnnSeqUpdate, fcSeqUpdate - flat, idxMap-packed gradient vectors
        %   totalAE, totalSE - sum of absolute and squared error over this pass, normalized by
        %                    numChannels
        %   totalSamples   - number of steps in this pass
        function [cnnSeqUpdate, rnnSeqUpdate, fcSeqUpdate, totalAE, totalSE, totalSamples] = backwardPass(obj, output, targetSequence)
            outputLength = size(output,1);

            totalAE      = 0;
            totalSE      = 0;
            totalSamples = 0;

            cnnSeqUpdate = zeros(obj.idxMap.cnnStrt(end)-1,1);
            fcSeqUpdate  = zeros(obj.idxMap.fcStrt(end)-1, 1);
            rnnSeqUpdate = zeros(obj.idxMap.rnn{end}(end,2),1);

            if ~isempty(obj.cnnModule)
                convIdx = 1;
                for i = numel(obj.cnnModule):-1:1
                    if isa(obj.cnnModule{i}, 'ConvolutionalLayer')
                        convIdx = i;
                    end
                end
                mapSize = size(obj.cnnModule{convIdx}.preactCache);
                dxCNN   = zeros([outputLength*obj.tPool, mapSize(2:end)]);
            end

            for scanningIdx = outputLength:-1:1
                winStart = (scanningIdx-1)*(obj.cnnStepSize*obj.tPool)+1;
                winEnd   = winStart+obj.cnnStepSize*(obj.tPool-1)+obj.cnnWindowSize-1;

                targetIdx = min(winEnd+1, size(targetSequence,1));
                target    = targetSequence(targetIdx, :);

                res = output(scanningIdx,:) - target;
                dx  = res;

                totalAE = totalAE + sum(abs(res))/obj.numChannels;
                totalSE = totalSE + sum(res.^2)/obj.numChannels;

                if ~isempty(obj.fcModule)
                    [dx, localFcUpdate] = obj.fcModule.backprop(dx, scanningIdx, obj.idxMap);
                end

                for i = numel(obj.rnnModule):-1:1
                    [dx, weightUpdate, biasUpdate] = obj.rnnModule{i}.backprop(dx, scanningIdx);
                    rnnStepUpdate = zeros(obj.idxMap.rnn{i}(end,2),1);
                    for j = 1:numel(weightUpdate)
                        strtIdx = obj.idxMap.rnn{i}(j,1);
                        endIdx  = obj.idxMap.rnn{i}(j,2);
                        rnnStepUpdate(strtIdx:endIdx) = [weightUpdate{j}(:); biasUpdate{j}(:)];
                    end

                    strtIdx = obj.idxMap.rnn{i}(1,  1);
                    endIdx  = obj.idxMap.rnn{i}(end,2);

                    rnnSeqUpdate(strtIdx:endIdx) = rnnSeqUpdate(strtIdx:endIdx) + rnnStepUpdate(strtIdx:endIdx);
                end

                if ~isempty(obj.cnnModule)
                    dx     = reshape(dx, [obj.tPool, mapSize(2:end)]);
                    startT = (scanningIdx-1)*obj.tPool + 1;
                    endT   = startT + obj.tPool - 1;
                    dxCNN(startT:endT, :, :) = dx;
                end

                fcSeqUpdate = fcSeqUpdate + localFcUpdate;
            end

            convLayerIdx = numel(obj.idxMap.cnnEnd);

            for i = numel(obj.cnnModule):-1:1
                [dxCNN, weightUpdate, biasUpdate] = obj.cnnModule{i}.backprop(dxCNN);
                if isa(obj.cnnModule{i}, "ConvolutionalLayer")
                    CNNLayerUpdate = [weightUpdate(:); biasUpdate(:)];
                    strtIdx = obj.idxMap.cnnStrt(convLayerIdx);
                    endIdx  = obj.idxMap.cnnEnd(convLayerIdx);
                    localCnnUpdate(strtIdx:endIdx) = CNNLayerUpdate;
                    convLayerIdx = convLayerIdx - 1;
                end
            end

            if ~isempty(obj.cnnModule)
                cnnSeqUpdate = cnnSeqUpdate + localCnnUpdate;
            end
            totalSamples = totalSamples + outputLength;
        end

        %% segmentSequences: one-step-shifted segment indices for forecasting
        % Splits each trial into one-step-shifted segments of length segmentLength + forecastLength,
        % to reserve headroom for train() to build its forecast.
        %
        % Inputs:
        %   trainingData   - N x 1 cell array, one row per trial.
        %   segmentLength  - length of each segment's context portion.
        %   forecastLength - trailing raw steps to reserve past each context segment
        %
        % Output:
        %   segmInfos - [M x 3]: [trialIdx, startIdx, endIdx] per segment, M = total windows
        %               across all trials
        function segmInfos = segmentSequences(trainingData, segmentLength, forecastLength)

            if nargin < 3 || isempty(forecastLength)
                forecastLength = 1;
            end
            numTrials = size(trainingData, 1);
 
            numWindowsPerTrial = zeros(numTrials, 1);
            for i = 1:numTrials
                rawLength = size(trainingData{i,1}, 1);
                numWindowsPerTrial(i) = max(rawLength - segmentLength - forecastLength + 1, 0);
            end
 
            totalRows = sum(numWindowsPerTrial);
 
            trialIdx = zeros(totalRows, 1);
            startIdx = zeros(totalRows, 1);
            endIdx   = zeros(totalRows, 1);
 
            rowPtr = 0;
            for i = 1:numTrials
                n = numWindowsPerTrial(i);
                if n == 0
                    continue;
                end
                rows = rowPtr+1 : rowPtr+n;
                trialIdx(rows) = i;
                startIdx(rows) = (1:n)';
                endIdx(rows)   = (1:n)' + segmentLength - 1;
                rowPtr = rowPtr + n;
            end
 
            segmInfos = [trialIdx, startIdx, endIdx];
        end
    end
end
