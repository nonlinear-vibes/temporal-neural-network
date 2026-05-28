classdef TemporalNeuralNet < handle
    %% TEMPORALNEURALNET: configurable CNN -> RNN/GRU/LSTM -> FC sequence classifier
    % 
    % A modular deep learning model for multivariate time series with dense sequence 
    % labeling, where each timestep has an associated one-hot target vector.
    % The pipeline is:
    %   1) 1-D temporal CNN module (Conv / Pool layers) for feature extraction
    %   2) Recurrent module (vanilla RNN, GRU, or LSTM) for temporal modelling
    %   3) Fully-connected classifier for final predictions
    %   All modules are optional, but the order CNN->RNN->FC is fixed.
    %
    % ───────────────────────────────────────────────────────────────────────────
    % KEY METHODS:
    %   net = TemporalNeuralNet(validationData, Name,Value, ...) 
    %       Constructs the network with the layer configurations specified by name-value pairs and
    %       initializes training metrics using the validation set (validationData).
    %
    %   networkOutput = net.forward(inputSequence)
    %       Runs inference on a raw input sequence (T×C). Returns per-timestep class probabilities 
    %       [numSteps×numClasses] after softmax normalization.
    %
    %   net.train(trainingData, validationData, epochs, batchSize, 'numSegments',S)
    %       Trains the network using overlapping input segments and Adam optimizer.
    %
    %   acc = net.evaluate(testData)
    %       Computes classification accuracy on a test dataset.
    %
    % ───────────────────────────────────────────────────────────────────────────
    % DATA FORMAT
    %   trainingData/validationData/testData: N×2 cell array. Each row:
    %     { sequence, labels }
    %       sequence : [T×C] double/single (time × numChannels)
    %       labels   : [T×K] one-hot (K = numClasses)
    %
    % ───────────────────────────────────────────────────────────────────────────
    % NAME–VALUE CONSTRUCTOR ARGS
    %   'CNN'    : cell array defining convolutional and pooling layers ({}):
    %                {'conv', numChannels, inFeatures, outFeatures, kernelSize}
    %                {'pool', poolingRatio}
    %   'RNN'    : cell array of recurrent specs ({}). Each spec is one of:
    %                {'rnn',  inDim, hiddenDim, outDim}
    %                {'gru',  inDim, outDim}
    %                {'lstm', inDim, outDim}
    %   'FC'     : specification for the fully connected classifier ({}):
    %                { [in, h1, ..., numClasses] }
    %   'tPool'  : number of consecutive temporal feature slices grouped and passed to
    %              the recurrent or fully connected module. (2)
    %   'numClasses' : number of output classes (second dimension of the final output) (16)
    %   'eta'    : base Adam learning rate (20)
    %   'learningRateDecay' : decay per epoch (0.95)
    %   'beta_1' : Adam β1 (0.90)
    %   'beta_2' : Adam β2 (0.999)
    %   'timeStep' : stride between evaluated timesteps when no recurrent module is present,
    %                used to reduce forward-pass computational cost (tPool)
    %
    %   Note: The input/output sizes of consecutive layers must match: 
    %         - In CNN layers, 'inFeatures' must be equal to the 'outFeatures' of the
    %           previous layer.
    %         - The input size of the recurrent or fully connected module following the 
    %           CNN module must be: inDim = numChannels * outFeatures * tPool
    %         - If there is no convolutional module, the input of the
    %           recurrent/fully connected module is: inDim = numChannels * tPool
    %
    %  Derived parameters:
    %  cnnWindowSize – raw timesteps covered by one CNN output step
    %  cnnStepSize   – raw step size of the CNN window (of size cnnWindowSize)
    %                   Receptive field per output step:
    %                   RF = cnnWindowSize + (tPool - 1)*cnnStepSize
    %  idxMap        - stores index boundaries used to split flattened accumulated
    %                  gradient vectors back into per-layer weight and bias tensors
    %
    % ───────────────────────────────────────────────────────────────────────────
    % USAGE EXAMPLE (CNN + LSTM + GRU + FC)
    %   convSpecs = { ...
    %   {'conv', 76, 1, 2, 3}, ...
    %   {'pool', 2}, ...
    %   {'conv', 76, 2, 4, 3}, ...
    %   {'pool', 2}, ...
    %   {'conv', 76, 4, 6, 3}, ...
    %   {'pool', 2}
    %   };
    %
    %   rnnSpecs = { ...
    %   {'lstm', 456, 228}, ...
    %   {'gru', 228, 114}};
    %
    %   fcSpecs = { [114, 16] };
    %
    %   net  = TemporalNeuralNet(validationData, 'CNN',convSpecs, 'RNN',rnnSpecs, 'FC',fcSpecs, ...
    %                            'tPool',1, 'numClasses',16, 'eta',10);

    %%
    properties
        cnnModule
        rnnModule
        fcModule

        tPool
        numClasses
        timeStep

        cnnWindowSize
        cnnStepSize
        idxMap
        
        eta
        beta_1
        beta_2
        epsilon
        learningRateDecay
        t
        
        learningHistory
        totalEntropyHistory
        trainingAccuracyHistory
    end
    
    %%
    methods
        %% TemporalNeuralNet constructor
        function obj = TemporalNeuralNet(validationData, varargin)

            p = inputParser;
            addParameter(p,'CNN',               {},   @(x) iscell(x));
            addParameter(p,'RNN',               {},   @(x) iscell(x));
            addParameter(p,'FC',                {},   @(x) iscell(x));
            addParameter(p,'tPool',             1,    @(x) isnumeric(x)&&isscalar(x));
            addParameter(p,'numClasses',        16,   @(x) isnumeric(x)&&isscalar(x));
            addParameter(p,'eta',               20,   @(x) isnumeric(x)&&isscalar(x));
            addParameter(p,'learningRateDecay', 0.95, @(x) isnumeric(x)&&isscalar(x));
            addParameter(p,'beta_1',            0.90, @(x) isnumeric(x)&&isscalar(x));
            addParameter(p,'beta_2',            0.999,@(x) isnumeric(x)&&isscalar(x));
            addParameter(p,'timeStep',          [],   @(x) isnumeric(x)&&isscalar(x));
            parse(p, varargin{:});

            % Hyperparams
            convSpecs      = p.Results.CNN;
            rnnSpecs       = p.Results.RNN;
            fcSpecs        = p.Results.FC;
            obj.tPool      = p.Results.tPool;
            obj.numClasses = p.Results.numClasses;
            if isempty(p.Results.timeStep)
                obj.timeStep = p.Results.tPool;
            elseif ~isempty(rnnSpecs)
                warning("Recurrent layer is present, setting 'timeStep' to %d.", obj.tPool);
                obj.timeStep = p.Results.tPool;
            else
                obj.timeStep = p.Results.timeStep;
            end

            % Build convolutional and pooling layers
            obj.cnnModule = cell(size(convSpecs));
            for i = 1:numel(convSpecs)
                spec = convSpecs{i};
                switch spec{1}
                    case 'conv'
                        obj.cnnModule{i} = ConvolutionalLayer(spec{2},spec{3},spec{4},spec{5});
                    case 'pool'
                        obj.cnnModule{i} = PoolingLayer(spec{2});
                    otherwise
                        error('Unknown conv spec: %s', spec{1});
                end
            end
        
            % Build recurrent layers
            obj.rnnModule = cell(size(rnnSpecs));
            for i = 1:numel(rnnSpecs)
                spec = rnnSpecs{i};
                switch spec{1}
                    case 'rnn'
                        obj.rnnModule{i} = RecurrentUnit(spec{2},spec{3},spec{4});
                    case 'lstm'
                        obj.rnnModule{i} = LSTMUnit(spec{2},spec{3});
                    case 'gru'
                        obj.rnnModule{i} = GRUnit(spec{2},spec{3});
                    otherwise
                        error('Unknown RNN spec: %s', spec{1});
                end
            end
        
            % Build fully connected classifier
            obj.fcModule = cell(size(fcSpecs{1}));
            if ~isempty(fcSpecs)
                if numel(fcSpecs{1}) < 2
                    error('fcSpecs must contain at least two sizes (input and output).');
                else
                    obj.fcModule = FullyConnectedNetwork(fcSpecs{1});
                end
            end
        
            % Derived window parameters
            % cnnWindowSize: number of raw timesteps covered by one RNN input frame (multiple timesteps pooled together)
            % cnnStepSize: downsampled step between successive RNN inputs (single timestep)
            obj.cnnWindowSize = 1;
            obj.cnnStepSize   = 1;
            for i = numel(obj.cnnModule):-1:1
                layer = obj.cnnModule{i};
                if isa(layer, 'ConvolutionalLayer')
                    obj.cnnWindowSize     = obj.cnnWindowSize + layer.kernelSize - 1;
                elseif isa(layer, 'PoolingLayer')
                    obj.cnnWindowSize     = obj.cnnWindowSize * layer.poolingRatio;
                    obj.cnnStepSize       = obj.cnnStepSize   * layer.poolingRatio;
                end
            end

            % Index map to slice gradient vectors
            obj.idxMap.cnnStrt = 1;
            obj.idxMap.cnnEnd  = [];
            obj.idxMap.fcStrt  = 1;
            obj.idxMap.fcEnd   = [];

            for i = 1:numel(obj.cnnModule)
                layer = obj.cnnModule{i};
                if isa(layer, "ConvolutionalLayer")
                    numParams = numel(layer.weights) + numel(layer.biases);
                    obj.idxMap.cnnEnd  = [obj.idxMap.cnnEnd;  obj.idxMap.cnnStrt(end) + numParams - 1];
                    obj.idxMap.cnnStrt = [obj.idxMap.cnnStrt; obj.idxMap.cnnStrt(end) + numParams];
                end
            end

            obj.idxMap.rnn = cell(numel(obj.rnnModule),1); % {[start, end]}

            if ~isempty(obj.rnnModule)
                obj.idxMap.rnn{1} = [];
                for i = 1:numel(obj.rnnModule)
                    layer = obj.rnnModule{i};
                    strtIdx = 1;
                    endIdx  = 0;
                    for j = 1:numel(layer.weights)
                        numParams = numel(layer.weights{j}) + numel(layer.biases{j});
                        endIdx    = endIdx + numParams;
                        obj.idxMap.rnn{i} = [obj.idxMap.rnn{i}; [strtIdx, endIdx]];
                        strtIdx = strtIdx + numParams;
                    end
                end
            end

            if ~isempty(obj.fcModule)
                for i = 1:numel(obj.fcModule.sizes)-1
                    numParams = numel(obj.fcModule.weights{i}) + numel(obj.fcModule.biases{i});
                    obj.idxMap.fcEnd  = [obj.idxMap.fcEnd;  obj.idxMap.fcStrt(end) + numParams - 1];
                    obj.idxMap.fcStrt = [obj.idxMap.fcStrt; obj.idxMap.fcStrt(end) + numParams];
                end
            end
        
            % Learning parameters
            obj.eta               = p.Results.eta;
            obj.learningRateDecay = p.Results.learningRateDecay;
            obj.beta_1            = p.Results.beta_1;
            obj.beta_2            = p.Results.beta_2;
            obj.epsilon           = 1e-8;   % Small number to avoid division-by-zero
            obj.t                 = 1;      % Adam time-step counter
        
            % Initialize learning metrics and history
            initialAcc       = evaluate(obj, validationData);
            % learningHistory columns: [batchSize, avgSegmLen, eta, valAccuracy, trainAccuracy, trainLoss, elapsedTime]
            obj.learningHistory      = [0, 0, 0, initialAcc, NaN, NaN, NaN];
            obj.totalEntropyHistory     = [];
            obj.trainingAccuracyHistory = [];
        end
    
        %% Forward: Execute the network on an input sequence
        function networkOutput = forward(obj, inputSequence)
            % Input:  inputSequence - [inputLength × numChannels] raw time-series
            % Output: networkOutput - [numSteps × numClasses] softmax probabilities per timestep
    
            % CNN (processes the whole sequence)
            for i = 1:numel(obj.cnnModule)
                inputSequence = obj.cnnModule{i}.forward(inputSequence);
            end
        
            % Number of RNN/FC steps
            if isempty(obj.rnnModule)
                stepSize = obj.timeStep;
            else
                stepSize = obj.tPool;
            end
            numSteps = floor((size(inputSequence,1)-obj.tPool)/stepSize) + 1;
            % floor((N - windowSize)/stepSize) + 1

            networkOutput = zeros(numSteps,obj.numClasses);
        
            % Process each timestep by RNN and FCL
            for j = 1:numSteps
                % Extract the j-th block of size [numTimeSegments × featureDim]
                startIdx = (j-1)*stepSize+1;
                block = inputSequence(startIdx:startIdx+obj.tPool-1,:);
                
                % Flatten into a row vector for RNN/FC input
                inputStep = reshape(block,1,[]);
        
                % RNN
                for i = 1:numel(obj.rnnModule)
                    inputStep = obj.rnnModule{i}.forward(inputStep, numSteps, 1);
                end
        
                % FCL
                if ~isempty(obj.fcModule)
                    inputStep = obj.fcModule.forward(inputStep, numSteps, j);
                end
        
                % Softmax
                networkOutput(j,:) = softmax(inputStep);
            end
        end
    
        %% Train: Optimize network
        function train(obj, trainingData, validationData, epochs, batchSize, varargin)
            % Inputs:
            %   trainingData   : cell array {sequence, labels}
            %   validationData : data for validation in each epoch
            %   epochs         : number of full data passes
            %   numSegments    : number of non-overlapping segments
            %   batchSize      : segments per update
            %   trains with no regard to temporal dependencies if no 'numSegments' is given (e.g. in case of no RNN module)

            p = inputParser;
            addParameter(p,'numSegments', [], @(x) isnumeric(x));
            parse(p, varargin{:});
            numSegments = p.Results.numSegments;

            % Compute label frequencies for weighted loss
            labelWeights = countTrainingLabels(trainingData, obj.numClasses);

            % Precompute segment info: [trialIdx, startIdx, endIdx]
            segmInfos = segmentSequences(obj, trainingData, numSegments);

            numConvLayers = numel(obj.cnnModule);
            numRecLayers  = numel(obj.rnnModule);

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
                    rows      = randIdxList((b-1)*batchSize+1:min(b*batchSize,numel(randIdxList)));
                    batchInfo = segmInfos(rows, :);

                    % Preallocate arrays to store gradient updates for each parallel worker separately
                    cnnUpdateSize = 0;
                    if ~isempty(obj.idxMap.cnnEnd)
                        cnnUpdateSize = obj.idxMap.cnnEnd(end);
                    end

                    fcUpdateSize = 0;
                    if ~isempty(obj.idxMap.fcEnd)
                        fcUpdateSize = obj.idxMap.fcEnd(end);
                    end

                    cnnBatchUpdate = zeros(cnnUpdateSize,1);
                    fcBatchUpdate  = zeros(fcUpdateSize, 1);
                    if ~isempty(obj.rnnModule)
                        rnnBatchUpdate = zeros(obj.idxMap.rnn{end}(end,2),1);
                    end
                                        
                    % Go through the segments
                    parfor segmIdx = 1:size(batchInfo,1)
                        
                        % Initialize temporary variables to suppress warnings
                        mapSz = [];
                        dxCNN = [];
                        localCnnUpdate = [];
                        localRnnUpdate = [];

                        % Create local cell arrays for the current worker's layers
                        if ~isempty(obj.cnnModule)
                            localCnnUpdate = zeros(obj.idxMap.cnnEnd(end),1);
                        end
                        if ~isempty(obj.rnnModule)
                            localRnnUpdate = zeros(obj.idxMap.rnn{end}(end,2),1);
                        end
                        localFcUpdate  = zeros(obj.idxMap.fcEnd(end), 1);
        
                        % reset hidden states and stored activations in the RNN and FC layers
                        obj.resetMemory();
        
                        tr       = batchInfo(segmIdx,1);
                        startIdx = batchInfo(segmIdx,2);
                        endIdx   = batchInfo(segmIdx,3);
        
                        trainingSegment = trainingData{tr,1}(startIdx:endIdx,:);
                        labels          = trainingData{tr,2}(startIdx:endIdx,:);
        
                        % CNN forward pass
                        for i = 1:numConvLayers
                            trainingSegment = obj.cnnModule{i}.forward(trainingSegment, 'train');
                        end
        
                        outputLength = floor(size(trainingSegment,1)/obj.tPool);
                        output       = zeros(outputLength, obj.numClasses);
        
                        % Forward pass through RNN and FC
                        for scanningIdx = 1:outputLength
                            startIdx   = (scanningIdx-1)*obj.tPool+1;
                            endIdx     = startIdx + obj.tPool-1;
                            inputSlice = reshape(trainingSegment(startIdx:endIdx,:,:), 1, []);
        
                            % RNN
                            for i = 1:numRecLayers
                                inputSlice = obj.rnnModule{i}.forward(inputSlice, outputLength, scanningIdx, 'train');
                            end
                            
                            % FC
                            if ~isempty(obj.fcModule)
                                inputSlice = obj.fcModule.forward(inputSlice, outputLength, scanningIdx, 'train');
                            end
        
                            % Softmax
                            output(scanningIdx,:) = softmax(inputSlice);
                        end
        
                        % Storage for backpropagated error at the CNN output
                        if ~isempty(obj.cnnModule)
                            convIdx = find(cellfun(@(L) isa(L,'ConvolutionalLayer'), obj.cnnModule), 1, 'last');
                            mapSz   = size(obj.cnnModule{convIdx}.preacts);  % [T_down x featDim1 x featDim2]
                            dxCNN   = zeros([outputLength*obj.tPool, mapSz(2:end)]);
                        end
                            
                        % Backward pass from softmax through FC, RNN, CNN
                        for scanningIdx = outputLength:-1:1
                            % Compute average ground-truth over window
                            startIdx = (scanningIdx-1)*(obj.cnnStepSize*obj.tPool)+1;
                            endIdx   = min(startIdx+obj.cnnStepSize*(obj.tPool-1)+obj.cnnWindowSize-1,size(labels,1));
                            avgLabel = sum(labels(startIdx:endIdx,:))/(endIdx-startIdx+1);
        
                            % Update correct counter for learning metrics
                            [~,  guess] = max(output(scanningIdx,:));
                            [~, actual] = max(avgLabel);
                            if guess == actual
                                correctCount_batch = correctCount_batch + 1;
                            end
        
                            % Softmax gradient (p - y)/freq
                            res    = output(scanningIdx,:)-avgLabel;
                            dx     = (res./labelWeights);
                            softm  = max(output(scanningIdx,:), obj.epsilon);
                            ResSum = -dot((avgLabel./labelWeights),log(softm));
                            
                            % Cross-entropy for monitoring
                            totalEntropy_batch = totalEntropy_batch + ResSum;
        
                            % Backprop through FC
                            if ~isempty(obj.fcModule)
                                [dx, localFcUpdate] = obj.fcModule.backprop(dx, scanningIdx, obj.idxMap);
                            end

                            % Backprop through RNN
                            for i = numRecLayers:-1:1
                                [dx, weightUpdate, biasUpdate] = obj.rnnModule{i}.backprop(dx, scanningIdx);
                                rnnLayerUpdate = zeros(obj.idxMap.rnn{i}(end,2),1);
                                for j = 1:numel(weightUpdate)
                                    strtIdx = obj.idxMap.rnn{i}(j,1);
                                    endIdx  = obj.idxMap.rnn{i}(j,2);
                                    rnnLayerUpdate(strtIdx:endIdx) = [weightUpdate{j}(:); biasUpdate{j}(:)];
                                end
                                % rnnLayerUpdate = [weightUpdate(:); biasUpdate(:)];
                                strtIdx = obj.idxMap.rnn{i}(1,  1);
                                endIdx  = obj.idxMap.rnn{i}(end,2);
                                localRnnUpdate(strtIdx:endIdx) = rnnLayerUpdate(strtIdx:endIdx);
                            end
        
                            % Reshape and accumulate CNN error      
                            if ~isempty(obj.cnnModule)
                                dx     = reshape(dx, [obj.tPool, mapSz(2:end)]);
                                startT = (scanningIdx-1)*obj.tPool + 1;
                                endT   = startT + obj.tPool - 1;
                                dxCNN(startT:endT, :, :) = dx;
                            end
                            
                            if ~isempty(obj.rnnModule)
                                rnnBatchUpdate = rnnBatchUpdate + localRnnUpdate;
                            end
                            fcBatchUpdate  = fcBatchUpdate  + localFcUpdate;
                        end

                        convLayerIdx = numel(obj.idxMap.cnnEnd);

                        for i = numConvLayers:-1:1
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
                            cnnBatchUpdate = cnnBatchUpdate + localCnnUpdate;
                        end
                        totalSamples_batch = totalSamples_batch + outputLength; 
                    end

                    % Reshape the gradient vectors and update the main object serially
                    convLayerIdx = 0;
                    for i = 1:numConvLayers
                        if isa(obj.cnnModule{i}, "ConvolutionalLayer")
                            convLayerIdx = convLayerIdx + 1;
                            strtIdx      = obj.idxMap.cnnStrt(convLayerIdx);
                            endIdx       = obj.idxMap.cnnEnd(convLayerIdx);
                            numBiases    = numel(obj.cnnModule{i}.biases);
                            obj.cnnModule{i}.dW = reshape(cnnBatchUpdate(strtIdx:endIdx-numBiases),  size(obj.cnnModule{i}.dW));
                            obj.cnnModule{i}.db = reshape(cnnBatchUpdate(endIdx-numBiases+1:endIdx), size(obj.cnnModule{i}.db));
                        end
                    end

                    for i = 1:numRecLayers
                        for j = 1:numel(obj.rnnModule{i}.weights)
                            strtIdx   = obj.idxMap.rnn{i}(j,1);
                            endIdx    = obj.idxMap.rnn{i}(j,2);
                            numBiases = numel(obj.rnnModule{i}.biases{j});
                            obj.rnnModule{i}.dW{j} = reshape(rnnBatchUpdate(strtIdx:endIdx-numBiases),  size(obj.rnnModule{i}.dW{j}));
                            obj.rnnModule{i}.db{j} = reshape(rnnBatchUpdate(endIdx-numBiases+1:endIdx), size(obj.rnnModule{i}.db{j}));
                        end
                    end

					if ~isempty(obj.fcModule)
					    for i = 1:numel(obj.fcModule.sizes)-1
						    strtIdx   = obj.idxMap.fcStrt(i);
                            endIdx    = obj.idxMap.fcEnd(i);
                            numBiases = numel(obj.fcModule.biases{i});
                            obj.fcModule.dW{i} = reshape(fcBatchUpdate(strtIdx:endIdx-numBiases),  size(obj.fcModule.dW{i}));
                            obj.fcModule.db{i} = reshape(fcBatchUpdate(endIdx-numBiases+1:endIdx), size(obj.fcModule.db{i}));
                        end
                    end
                    
        
                    % Adam parameter update
                    adamOptimizer(obj, totalSamples_batch);
                    obj.t = obj.t + 1;
        
                    % Record batch metrics
                    obj.totalEntropyHistory     = [obj.totalEntropyHistory;     totalEntropy_batch/totalSamples_batch];
                    obj.trainingAccuracyHistory = [obj.trainingAccuracyHistory; correctCount_batch/totalSamples_batch];
        
                    % Plot progress
                    subplot(2,1,1); cla;
                    plot(obj.totalEntropyHistory);
                    grid on; % axis padded;
                    title('Training loss (avg cross-entropy)');
                    xlabel('Batch'); ylabel('Loss');

                    subplot(2,1,2); cla;
                    plot(obj.trainingAccuracyHistory);
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
                filename  = sprintf('Network_trained_%s.mat', timestamp);
                save(filename, 'obj');
            end
        end

        %% Evaluate: Compute overall classification accuracy on test set
        function acc = evaluate(obj, testData)
            % Inputs: 
            %   obj            - TemporalNeuralNet instance
            %   testData - N×2 cell array, each row: {rawSequence, oneHotLabels}
            % Output:
            %   acc            - fraction of correctly predicted time‐segments

            % Initialize counters
            correctCount = 0;
            totalSteps  = 0;
        
            % Loop through each item in the test data
            for i = 1:size(testData, 1)
                input  = testData{i,1};    % [T_raw × numChannels]
                labels = testData{i,2};    % [T_raw × numClasses]
                obj.resetMemory();         % clear any RNN/FC hidden state
        
                % Forward-pass through full network
                outputActivations = obj.forward(input);
        
                % Determine the predicted class
                [~, predictedClasses] = max(outputActivations,[],2);
                outputLength          = numel(predictedClasses);
                
                % For each segment, derive the “true” class at the segment midpoint
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

        %% adamOptimizer: caller function of 'applyAdam' in each layer 
        function adamOptimizer(obj, totalSamples_batch)
            % Apply Adam parameter updates to all trainable modules
            % totalSamples_batch: normalizing factor (batch size)

            for L = obj.cnnModule
                if isa(L, "ConvolutionalLayer")
                    L{1}.applyAdam(obj.eta, obj.beta_1, obj.beta_2, totalSamples_batch, obj.t, obj.epsilon);
                end
            end
            for U = obj.rnnModule
                U{1}.applyAdam(obj.eta, obj.beta_1, obj.beta_2, totalSamples_batch, obj.t, obj.epsilon);
            end
            if ~isempty(obj.fcModule)
                obj.fcModule.applyAdam(obj.eta, obj.beta_1, obj.beta_2, totalSamples_batch, obj.t, obj.epsilon);
            end
        end

        %% segmentSequences: generate overlapping segment indices
        function segmInfos = segmentSequences(obj, trainingData, numSegments)
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
        
        %% resetMemory: clear stored activations and hidden states before each new sequence
        function resetMemory(obj)
            % Reset CNN activations
            for i = 1:numel(obj.cnnModule)
                obj.cnnModule{i}.resetStoredActivations();
            end
            % Reset each RNN unit's hidden state history
            for i = 1:numel(obj.rnnModule)
                obj.rnnModule{i}.resetMemory();
            end
            % Reset FC module if present (clears any stored activations)
            if ~isempty(obj.fcModule)
                obj.fcModule.resetStoredActivations();
            end
        end

        %% resetGrads: zero out gradient accumulators before each batch
        function resetGrads(obj)
            % Reset CNN gradient buffers
            for i = 1:numel(obj.cnnModule)/2 % build check fot layer type
                obj.cnnModule{(i-1)*2+1}.resetGrads();
            end
            % Reset RNN gradient accumulators
            for i = 1:numel(obj.rnnModule)
                obj.rnnModule{i}.resetGrads();
            end
            % Reset FC gradients if module exists
            if ~isempty(obj.fcModule)
                obj.fcModule.resetGrads();
            end
        end

    end
end

%% Softmax
function s = softmax(a)
    exp_a = exp(a-max(a));
    s     = exp_a / sum(exp_a);
end

%% Compute label weights based on their frequencies
function labelWeights = countTrainingLabels(trainingData, numClasses)

    labelCount = zeros(1,numClasses);
    
    for i = 1:size(trainingData,1)
        labelCount = labelCount + sum(trainingData{i,2});
    end
    
    labelFreqs = labelCount/sum(labelCount);
    labelFreqs(labelFreqs == 0) = 1;       % replace zeros with 1
    
    labelWeights = labelFreqs*numClasses;
end