classdef Predictor < TemporalNeuralNet
    %% PREDICTOR: forecasting subclass of TemporalNeuralNet
    %
    % Implements the sequence-forecasting specifics: autoregressive
    % multi-step forward, teacher-forced training on one-step-shifted
    % windows, MSE loss/backprop, and MSE evaluation.
    %
    % See TemporalNeuralNet.m for the shared CNN->RNN->FC backbone,
    % Adam optimizer, and memory/gradient reset helpers -- forwardStep,
    % adamOptimizer, resetMemory, and resetGrads are all inherited and
    % NOT redefined here.
 
    properties
        forecastLength   % default horizon
        trainingLength
    end
 
    methods
        %% Predictor constructor
        function obj = Predictor(validationData, varargin)
            p = inputParser;
            p.KeepUnmatched = true;   % everything else forwards to the parent
            addParameter(p,'forecastLength', 12, @(x) isnumeric(x)&&isscalar(x));
            addParameter(p,'segmentLength', 24, @(x) isnumeric(x)&&isscalar(x));
            parse(p, varargin{:});
 
            % Forward CNN/RNN/FC/tPool/numChannels/eta/... to the shared parent constructor.
            names = fieldnames(p.Unmatched);
            vals  = struct2cell(p.Unmatched);
            unmatchedArgs = [names vals]';
            unmatchedArgs = unmatchedArgs(:)';
            obj@TemporalNeuralNet(unmatchedArgs{:});
 
            obj.forecastLength = p.Results.forecastLength;
            obj.segmentLength  = p.Results.segmentLength;
            obj.outputWidth    = obj.numChannels;
 
            initialMSE = obj.evaluate(validationData, obj.forecastLength, obj.segmentLength);
            obj.learningHistory = [0, obj.forecastLength, 0, initialMSE, NaN, NaN, NaN];
        end
 
        %% outputActivation: no activation on the raw FC output
        function a = outputActivation(~, z)
            a = z;
        end
 
        %% outputWidth: regression maps channels -> channels
        function w = outputWidth(obj)
            w = obj.numChannels;
        end
 
        %% Forward: Autoregressive multi-step forecast (single sequence or batch)
        function networkOutput = forecast(obj, inputData, varargin)
            p = inputParser;
            addParameter(p,'isTraining',     false,@(x) islogical(x));
            addParameter(p,'forecastLength', 1,    @(x) isnumeric(x)&&isscalar(x));
            addParameter(p,'segmLength',     NaN,  @(x) isnumeric(x)&&isscalar(x));
            parse(p, varargin{:});
            isTraining          = p.Results.isTraining;
            forecastLengthLocal = p.Results.forecastLength;
            segmLength          = p.Results.segmLength;
 
            wasCell = iscell(inputData);
            if wasCell
                sequences = inputData(:,1);
            else
                sequences = {inputData};
            end
 
            outputs = cell(numel(sequences),1);
            for s = 1:numel(sequences)
                outputs{s} = obj.forecastSequence(sequences{s}, isTraining, forecastLengthLocal, segmLength);
            end
 
            if wasCell
                networkOutput = outputs;
            else
                networkOutput = outputs{1};
            end
        end
 
        %% forecastSequence: autoregressive rollout for a single raw sequence
        function networkOutput = forecastSequence(obj, inputSequence, isTraining, forecastLengthLocal, segmLength)
            T = size(inputSequence,1);
 
            if isnan(segmLength)
                segmLength = T - forecastLengthLocal;
            end
 
            startIdx = T - forecastLengthLocal - segmLength + 1;
            endIdx   = T - forecastLengthLocal;
            if startIdx < 1
                error('forward:insufficientLength', ...
                    ['inputSequence (length %d) is too short for ' ...
                     'segmLength (%d) and forecastLength (%d).'], ...
                     T, segmLength, forecastLengthLocal);
            end
 
            networkOutput = inputSequence;
            window        = inputSequence(startIdx:endIdx, :);
 
            for k = 1:forecastLengthLocal
                obj.resetMemory();
                stepOutput = obj.forward(window, isTraining=isTraining);
                prediction = stepOutput(end, :);
 
                window = [window(2:end, :); prediction];
                networkOutput(T-forecastLengthLocal+k, :) = prediction;
            end
        end
 
        %% Train: Optimize network via teacher-forced next-step prediction
        function train(obj, trainingData, validationData, epochs, batchSize, varargin)
            p = inputParser;
            addParameter(p,'segmentLength',  [], @(x) isnumeric(x));
            parse(p, varargin{:});
            segmentLength = p.Results.segmentLength;
            forecastLengthLocal = 1;
 
            segmInfos = obj.segmentSequences(trainingData, segmentLength);
 
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
                        batchData{i} = trainingData{trialIdx,1}(startIdx:endIdx+forecastLengthLocal,:);
                    end
 
                    parfor segmIdx = 1:size(trialIdxs,1)
 
                        obj.resetMemory();
                        trainSegment   = batchData{segmIdx};
                        contextSegment = trainSegment(1:end-forecastLengthLocal, :);
 
                        output = obj.forward(contextSegment, 'isTraining', true);
 
                        [cnnSeqUpdate, rnnSeqUpdate, fcSeqUpdate, totalAE, totalSE, totalSamples] = obj.backwardPass(output, trainSegment);
 
                        cnnBatchUpdate = cnnBatchUpdate + cnnSeqUpdate;
                        rnnBatchUpdate = rnnBatchUpdate + rnnSeqUpdate;
                        fcBatchUpdate  = fcBatchUpdate  + fcSeqUpdate;
 
                        totalAE_batch      = totalAE_batch      + totalAE;
                        totalSE_batch      = totalSE_batch      + totalSE;
                        totalSamples_batch = totalSamples_batch + totalSamples;
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
                    grid on; % axis padded;
                    title('Training loss (MSE)');
                    xlabel('Batch'); ylabel('MSE');

                    subplot(2,1,2); cla;
                    plot(obj.trainingMetricHistory);
                    grid on; % axis padded;
                    title('Validation metric (MAE)');
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
                validationMAE = obj.evaluate(validationData, forecastLengthLocal, segmentLength);
                obj.learningHistory(end+1,:) = [batchSize, forecastLengthLocal, obj.eta, validationMAE, trainingMAE, residual, elapsedTime];
                fprintf('Validation MAE: %.4f\n', validationMAE);
 
                obj.eta = obj.learningRateDecay*obj.eta;
 
                timestamp = datetime('now', 'Format','yyyy-MM-dd_HH-mm-SS');
                filename  = sprintf('Predictor_trained_%s.mat', timestamp);
                save(filename, 'obj');
            end
        end
 
        %% Evaluate: Compute average forecasting MAE over a dataset
        function MAE = evaluate(obj, data, forecastLength, segmentLength)
            n = size(data, 1);
            totalSE    = 0;
            totalSteps = 0;
 
            for i = 1:n
                input = data{i,1};
 
                if size(input,1) < forecastLength + segmentLength
                    continue;
                end
 
                obj.resetMemory();
 
                output = obj.forecast(input, forecastLength=forecastLength, segmLength=segmentLength);
 
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
    end
 
    methods (Static)
        %% segmentSequences: one-step-shifted segment indices for forecasting
        function segmInfos = segmentSequences(trainingData, segmentLength)
            numTrials = size(trainingData, 1);
 
            numWindowsPerTrial = zeros(numTrials, 1);
            for i = 1:numTrials
                rawLength = size(trainingData{i,1}, 1);
                numWindowsPerTrial(i) = max(rawLength - segmentLength, 0);
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
