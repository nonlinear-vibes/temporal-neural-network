classdef Classifier < TemporalNeuralNet
    %% CLASSIFIER: dense sequence-labeling subclass of TemporalNeuralNet
    %
 
    properties
        numClasses
    end
 
    methods
        %% Classifier constructor
        function obj = Classifier(trainingData, varargin)
            p = inputParser;
            p.KeepUnmatched = true;
            addParameter(p,'numClasses', 16, @(x) isnumeric(x)&&isscalar(x));
            parse(p, varargin{:});
 
            names = fieldnames(p.Unmatched);
            vals  = struct2cell(p.Unmatched);
            unmatchedArgs = [names vals]';
            unmatchedArgs = unmatchedArgs(:)';
            obj@TemporalNeuralNet(unmatchedArgs{:});
 
            obj.outputWidth = p.Results.numClasses;
 
            initialAcc      = evaluate(obj, trainingData);
            % learningHistory columns: [batchSize, avgSegmLen, eta, valAccuracy, trainAccuracy, trainLoss, elapsedTime]
            obj.learningHistory     = [0, 0, 0, initialAcc, NaN, NaN, NaN];
        end
 
        %% outputActivation: classification -- softmax over the raw FC output
        function a = outputActivation(~, z)
            a = softmx(z);
        end
 
        %% Train: Optimize network
        function train(obj, trainingData, validationData, epochs, batchSize, varargin)
            % Inputs:
            %   trainingData   - cell array {sequence, labels}
            %   validationData - data for validation in each epoch
            %   epochs         - number of full data passes
            %   batchSize      - segments per update
            %
            %   trains with no regard to temporal dependencies if no 'numSegments' is given
            %   (e.g. in case of no RNN module)

            p = inputParser;
            addParameter(p,'numSegments', [], @(x) isnumeric(x)); % number of non-overlapping segments
            parse(p, varargin{:});
            numSegments  = p.Results.numSegments;

            % Compute label frequencies for weighted loss
            labelWeights = countTrainingLabels(trainingData, obj.outputWidth);

            % Precompute segment info: [trialIdx, startIdx, endIdx]
            segmInfos    = obj.segmentSequences(trainingData, numSegments);

            % Go through the epochs
            for epochIdx = 1:epochs
                tic

                % Shuffle segments
                randIdxList = randperm(size(segmInfos,1));

                % Epoch counters
                correctCount_epoch = 0;
                totalEntropy_epoch = 0;
                totalSamples_epoch = 0;
        
                % Go through the batches
                for b = 1:ceil(size(segmInfos,1)/batchSize)
        
                    % Clear gradient accumulators
                    obj.resetGrads();
                    
                    % Batch counters
                    correctCount_batch = 0;
                    totalEntropy_batch = 0;
                    totalSamples_batch = 0;
        
                    % List of segments in the batch
                    rows           = randIdxList((b-1)*batchSize+1:min(b*batchSize,numel(randIdxList)));
                    trialIdxs      = segmInfos(rows, 1);
                    startIdxs      = segmInfos(rows, 2);
                    endIdxs        = segmInfos(rows, 3);

                    % Preallocate arrays to store gradient updates for each parallel worker separately
                    cnnUpdateSize  = obj.idxMap.cnnStrt(end)-1;
                    fcUpdateSize   = obj.idxMap.fcStrt(end)-1;

                    cnnBatchUpdate = zeros(cnnUpdateSize,1);
                    fcBatchUpdate  = zeros(fcUpdateSize, 1);
                    rnnBatchUpdate = zeros(obj.idxMap.rnn{end}(end,2),1);

                    batchData      = cell(numel(rows),1);
                    batchLbls      = cell(numel(rows),1);

                    for i = 1:numel(rows)
                        trialIdx     = trialIdxs(i);
                        startIdx     = startIdxs(i);
                        endIdx       = endIdxs(i);
                        batchData{i} = trainingData{trialIdx,1}(startIdx:endIdx,:);
                        batchLbls{i} = trainingData{trialIdx,2}(startIdx:endIdx,:);
                    end
                                        
                    % Go through the segments
                    parfor segmIdx = 1:size(trialIdxs,1)
        
                        % Reset hidden states and stored activations in the RNN and FC layers
                        obj.resetMemory();
                        trainSegment       = batchData{segmIdx};
                        labels             = batchLbls{segmIdx};
        
                        % Forward pass with activation and preactivation storage enabled
                        output             = obj.forward(trainSegment, 'isTraining',true);
                            
                        % Backward pass from softmax through FC, RNN, CNN
                        [cnnSeqUpdate, rnnSeqUpdate, fcSeqUpdate, correctCount, totalEntropy, totalSamples] = obj.backwardPass(output, labels, labelWeights);

                        cnnBatchUpdate     = cnnBatchUpdate + cnnSeqUpdate;
                        rnnBatchUpdate     = rnnBatchUpdate + rnnSeqUpdate;
                        fcBatchUpdate      = fcBatchUpdate  + fcSeqUpdate;

                        correctCount_batch = correctCount_batch + correctCount;
                        totalEntropy_batch = totalEntropy_batch + totalEntropy;
                        totalSamples_batch = totalSamples_batch + totalSamples;
                    end

                    % Reshape the gradient vectors and update the main object serially
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
        
                    % Adam parameter update
                    adamOptimizer(obj, totalSamples_batch);
                    obj.t = obj.t + 1;
        
                    % Record batch metrics
                    obj.totalLossHistory        = [obj.totalLossHistory;     totalEntropy_batch/totalSamples_batch];
                    obj.trainingMetricHistory = [obj.trainingMetricHistory; correctCount_batch/totalSamples_batch];
        
                    % Plot progress
                    subplot(2,1,1); cla;
                    plot(obj.totalLossHistory);
                    grid on; % axis padded;
                    title('Training loss (avg cross-entropy)');
                    xlabel('Batch'); ylabel('Loss');

                    subplot(2,1,2); cla;
                    plot(obj.trainingMetricHistory);
                    grid on; % axis padded;
                    title('Training accuracy');
                    xlabel('Batch'); ylabel('Accuracy');

                    drawnow
        
                    % Update epoch totals
                    correctCount_epoch = correctCount_epoch + correctCount_batch;
                    totalEntropy_epoch = totalEntropy_epoch + totalEntropy_batch;
                    totalSamples_epoch = totalSamples_epoch + totalSamples_batch;
                end
        
                elapsedTime = toc;
                fprintf('Epoch %d completed in %.0f s.\n', size(obj.learningHistory,1), elapsedTime);
        
                % Update learningHistory log
                trainingAccuracy = correctCount_epoch/totalSamples_epoch;
                residual         = totalEntropy_epoch/totalSamples_epoch;
                testAcc          = evaluate(obj, validationData);
                avgSegmLen       = totalSamples_epoch/size(segmInfos,1);
                obj.learningHistory(end+1,:) = [batchSize, avgSegmLen, obj.eta, testAcc, trainingAccuracy, residual, elapsedTime];
                fprintf('Validation accuracy: %.4f\n', testAcc);
                % Learning rate decay
                obj.eta = obj.learningRateDecay*obj.eta;
        
                % Save trained network
                timestamp = datetime('now', 'Format','yyyy-MM-dd_HH-mm-SS');
                filename  = sprintf('Classifier_trained_%s.mat', timestamp);
                save(filename, 'obj');
            end
        end
 
        %% Evaluate: Compute overall classification accuracy on test set
        function acc = evaluate(obj, data)
            % Inputs: 
            %   testData  - N×2 cell array, each row: {rawSequence, oneHotLabels}
            % Output:
            %   acc       - fraction of correctly predicted time‐segments

            % Initialize counters
            correctCount = 0;
            totalSteps  = 0;
        
            % Loop through each item in the test data
            for i = 1:size(data, 1)
                input  = data{i,1};    % [T_raw × numChannels]
                labels = data{i,2};    % [T_raw × numClasses]
                obj.resetMemory();     % clear any RNN/FC hidden state
        
                % Forward-pass through full network
                outputActivations = obj.forward(input);
        
                % Determine the predicted class
                [~, predictedClasses] = max(outputActivations,[],2);
                outputLength          = numel(predictedClasses);
                
                % For each segment, derive the "true" class at the segment midpoint
                for j = 1:outputLength
                    % Compute raw‐data window corresponding to segment t
                    if isempty(obj.rnnModule)
                        stepSize = obj.timeStep;
                    else
                        stepSize = obj.tPool;
                    end

                    startIdx = (j-1)*(obj.cnnStepSize*stepSize)+1;
                    endIdx   = min(startIdx+obj.cnnStepSize*(obj.tPool-1)+obj.cnnWindowSize-1,size(labels,1));
                    % During evaluation, segment labels are assigned using the midpoint timestep
                    midpoint = floor((startIdx + endIdx)/2);

                    % True class at midpoint
                    [~, trueClass] = max(labels(midpoint,:), [], 2);

                    % Compare prediction with truth
                    if predictedClasses(j) == trueClass
                        correctCount = correctCount + 1;
                    end
                end
        
                % Accumulate count of evaluated segments
                totalSteps = totalSteps + outputLength;
                
            end
        
            % Overall success rate
            if totalSteps == 0
                warning('No evaluable frames found.');
                acc = NaN;
            else
                acc = correctCount/totalSteps;
            end
        end
 
        %% backwardPass: runs backpropagation through all the layers of the network
        function [cnnSeqUpdate, rnnSeqUpdate, fcSeqUpdate, correctCount, totalEntropy, totalSamples] = backwardPass(obj, output, labels, labelWeights)
            % Inputs:
            %   output       - Final classification output of the network, [numSteps × numClasses]
            %   labels       - One-hot encoded ground truth labels, [numSteps × numClasses]
            %   labelWeights - Frequency of labels, [1 × numClasses]
            % Outputs:
            %   cnnSeqUpdate - Gradient update for the CNN module
            %   rnnSeqUpdate - Gradient update for the RNN module
            %   fcSeqUpdate  - Gradient update for the FC module
            %   correctCount - Number of correctly classified steps
            %   totalEntropy - Sum of the residuals over the whole sequence
            %   totalSamples - Number of steps in the sequence

            outputLength   = size(output,1);

            % Local counters
            correctCount   = 0;
            totalEntropy   = 0;
            totalSamples   = 0;

            % Storage for weight and bias updates
            cnnSeqUpdate   = zeros(obj.idxMap.cnnEnd(end),1);
            fcSeqUpdate    = zeros(obj.idxMap.fcEnd(end), 1);
            rnnSeqUpdate   = zeros(obj.idxMap.rnn{end}(end,2),1);

            % Storage for backpropagated error at the CNN output
            if ~isempty(obj.cnnModule)
                convIdx = 1;
                for i = numel(obj.cnnModule):-1:1
                    if isa(obj.cnnModule, 'ConvolutionalLayer')
                        convIdx = i;
                    end
                end
                mapSize = size(obj.cnnModule{convIdx}.preactCache);  % [T_down x featDim1 x featDim2]
                dxCNN   = zeros([outputLength*obj.tPool, mapSize(2:end)]);
            end

            for scanningIdx = outputLength:-1:1
                % Compute average ground-truth over window
                startIdx = (scanningIdx-1)*(obj.cnnStepSize*obj.tPool)+1;
                endIdx   = min(startIdx+obj.cnnStepSize*(obj.tPool-1)+obj.cnnWindowSize-1,size(labels,1));
                avgLabel = sum(labels(startIdx:endIdx,:))/(endIdx-startIdx+1);

                % Update correct counter for learning metrics
                [~,  guess] = max(output(scanningIdx,:));
                [~, actual] = max(avgLabel);
                if guess == actual
                    correctCount = correctCount + 1;
                end

                % Softmax gradient (p - y)/freq
                res    = output(scanningIdx,:)-avgLabel;
                dx     = (res./labelWeights);
                softm  = max(output(scanningIdx,:), obj.epsilon);
                resSum = -dot((avgLabel./labelWeights),log(softm));
                
                % Cross-entropy for monitoring
                totalEntropy = totalEntropy + resSum;

                % Backprop through FC
                if ~isempty(obj.fcModule)
                    [dx, localFcUpdate] = obj.fcModule.backprop(dx, scanningIdx, obj.idxMap);
                end

                % Backprop through RNN
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

                % Reshape and accumulate CNN error      
                if ~isempty(obj.cnnModule)
                    dx     = reshape(dx, [obj.tPool, mapSize(2:end)]);
                    startT = (scanningIdx-1)*obj.tPool + 1;
                    endT   = startT + obj.tPool - 1;
                    dxCNN(startT:endT, :, :) = dx;
                end
                
                fcSeqUpdate  = fcSeqUpdate  + localFcUpdate;
            end

            % Backprop through the CNN module
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
        %% segmentSequences: overlapping fixed-count segments per trial
        function segmInfos = segmentSequences(trainingData, numSegments)
            % Splits each trial into overlapping windows.
            % Inputs:
            %   trainingData: N×2 cell array {data, labels}
            %   numSegments : number of base segments per trial
            % Output:
            %   segmInfos   : M×3 array [trialIdx, startIdx, endIdx]
        
            if ~isempty(numSegments)
                % Calculate total windows per trial with 75% overlap
                segmPerTrial = numSegments + 3*(numSegments-1);
                numTrials    = size(trainingData, 1);
                totalRows    = numTrials*segmPerTrial;
            
                % Preallocate index arrays
                trialIdx = zeros(totalRows,1);
                startIdx = zeros(totalRows,1);
                endIdx   = zeros(totalRows,1);
            
                % Compute start and end indices of each segment
                for i = 1:numTrials
                    rawLength  = size(trainingData{i,1},1);
                    segmLength = floor(rawLength/numSegments);
                    shift      = floor(segmLength/4);
                    base       = (i-1)*segmPerTrial+1;
    
                    trialIdx(base:i*segmPerTrial) = ones(segmPerTrial,1)*i;
                    startIdx(base:i*segmPerTrial) = [0:segmPerTrial-1]*shift+1;
                    endIdx(base:i*segmPerTrial)   = [0:segmPerTrial-1]*shift+segmLength;
                end

            else
                numTrials  = size(trainingData,1);
                trialIdx   = [];
                startIdx   = [];
                endIdx     = [];
                windowSize = (obj.cnnWindowSize+(obj.tPool-1)*obj.cnnStepSize);
                shift      = 2;

                for i = 1:numTrials
                    trialLength = size(trainingData{i},1);
                    numWindows  = floor((trialLength-windowSize)/2)+1;

                    trialIdx(end+1:end+numWindows,1) = ones(numWindows,1)*i;
                    startIdx(end+1:end+numWindows,1) = [0:numWindows-1]*shift+1;
                    endIdx(end+1:end+numWindows,1)   = [0:numWindows-1]*shift+windowSize;
                end
            end
            % Combine into final M×3 matrix
            segmInfos = [trialIdx, startIdx, endIdx];
        end
    end
end

%% Compute label weights based on their frequencies
function labelWeights = countTrainingLabels(trainingData, numClasses)

    labelCount = zeros(1,numClasses);
    
    for i = 1:size(trainingData,1)
        labelCount = labelCount + sum(trainingData{i,2});
    end
    
    labelFreqs = labelCount/sum(labelCount);
    labelFreqs(labelFreqs == 0) = 1;       % replace zeroes with 1
    
    labelWeights = labelFreqs*numClasses;
end

%% Softmax
function s = softmx(a)
    exp_a = exp(a-max(a));
    s     = exp_a / sum(exp_a);
end