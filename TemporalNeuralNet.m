classdef TemporalNeuralNet < handle
    %% TEMPORALNEURALNET: configurable CNN → RNN/GRU/LSTM → FC sequence classifier
    % 
    % A modular deep learning model for multivariate time series. The pipeline is:
    %   1) 1-D temporal CNN module (Conv / Pool) for feature extraction
    %   2) Recurrent module (vanilla RNN, GRU, or LSTM) for temporal modelling
    %   3) Fully-connected classifier for final predictions
    %   All modules are optional, but the order CNN->RNN->FC is fixed.
    %
    % ───────────────────────────────────────────────────────────────────────────
    % KEY METHODS:
    %   net = TemporalNeuralNet(testData, Name, Value, ...) 
    %       Constructs the network with the layer configurations specified by name-value pairs and
    %       initializes training metrics using the validation set (testData).
    %
    %   networkOutput = forward(obj, inputSequence)
    %       Runs inference on a raw input sequence (T×C). Returns [numSteps×numClasses]
    %       softmax probabilities.
    %
    %   train(obj, trainingData, testData, epochs, batchSize, 'numSegments',S)
    %       Trains with overlapping segments and Adam.
    %
    %   acc = evaluate(obj, testData)
    %       Computes classification accuracy on a test dataset.
    %
    %   segmInfos = segmentSequences(~, trainingData, numSegments)
    %       Generates overlapping segment indices for training.
    %
    %   resetMemory(obj) / resetGrads(obj) / adamOptimizer(obj, m)
    %       Clears recurrent and any stored activation states before a new sequence.
    %       Zeroes gradient accumulators before each batch.
    %       Applies Adam updates to all network parameters.
    %
    % ───────────────────────────────────────────────────────────────────────────
    % DATA FORMAT
    %   trainingData/testData: N×2 cell array. Each row:
    %     { sequence, labels }
    %       sequence : [T×C] double/single (time × channels)
    %       labels   : [T×K] one-hot (K = numClasses)
    %
    % ───────────────────────────────────────────────────────────────────────────
    % NAME–VALUE CONSTRUCTOR ARGS
    %   'CNN'    : cell array defining convolutional and pooling layers ({}):
    %                {'conv', numChannels, inputLayers, numKernels, kernelSize}
    %                {'pool', poolingRatio}
    %                note: 'conv' must always be followed by 'pool', but 'poolingRatio = 1' can be given
    %   'RNN'    : cell array of recurrent specs ({}). Each spec is one of:
    %                {'rnn',  inDim, hiddenDim, outDim}
    %                {'gru',  inDim, outDim}
    %                {'lstm', inDim, outDim}
    %   'FC'     : specification for the fully connected classifier ({}):
    %                { [in, h1, ..., numClasses] }
    %   'tPool'  : length of time segments received by the RNN/FC (2)
    %   'numClasses' : number of output classes (size of final output) (16)
    %   'eta'    : base Adam learning rate (20)
    %   'learningRateDecay' : decay per epoch (0.95)
    %   'beta_1' : Adam β1 (0.90)
    %   'beta_2' : Adam β2 (0.999)
    %   'timeStep' : integer step size used when NO RNN stack is present (1)
    %
    %   Note: The input/output sizes of consecutive layers must match: 
    %         - In CNN layers, 'inputLayers' must be eual to the 'numKernels' of the
    %           previous layer.
    %         - The input size of the recurrent or fully connected module following the 
    %           CNN module must be: inDim = numChannels * numKernels * tPool
    %         - If there is no convolutional module, the input of the
    %           recurrent/fully connected module is: inDim = numChannels * tPool
    %
    %           Derived parameters:
    %           CNNwindowSize – raw timesteps covered by one CNN output step
    %           CNNstepSize   – raw step size of the CNN window (of size CNNwindowSize)
    %           Receptive field per output step:
    %               RF = CNNwindowSize + (tPool − 1)*CNNstepSize
    %
    % ───────────────────────────────────────────────────────────────────────────
    % USAGE EXAMPLE
    %   % CNN + LSTM + FC
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
    %   net  = TemporalNeuralNet(testData, 'CNN',convSpecs, 'RNN',rnnSpecs, 'FC',fcSpecs, ...
    %                            'tPool',1, 'numClasses',16, 'eta',10);

    %%
    properties
        CNNmodule
        RNNmodule
        FCmodule

        tPool
        numClasses

        CNNwindowSize
        CNNstepSize
        timeStep
        
        eta
        beta_1
        beta_2
        epsilon
        t
        learningRateDecay

        learningHistory
        totalEntropy_vec
        trainingaccuracy_vec
    end
    
    %%
    methods
        %% TemporalNeuralNet constructor
        function obj = TemporalNeuralNet(testData, varargin)

            p = inputParser;
            addParameter(p,'CNN',               {},   @(x) iscell(x));
            addParameter(p,'RNN',               {},   @(x) iscell(x));
            addParameter(p,'FC',                {},   @(x) iscell(x));
            addParameter(p,'tPool',             2,    @(x) isnumeric(x)&&isscalar(x));
            addParameter(p,'numClasses',        16,   @(x) isnumeric(x)&&isscalar(x));
            addParameter(p,'eta',               20,  @(x) isnumeric(x)&&isscalar(x));
            addParameter(p,'learningRateDecay', 0.95, @(x) isnumeric(x)&&isscalar(x));
            addParameter(p,'beta_1',            0.90, @(x) isnumeric(x)&&isscalar(x));
            addParameter(p,'beta_2',            0.999,@(x) isnumeric(x)&&isscalar(x));
            addParameter(p,'timeStep',          1,    @(x) isnumeric(x)&&isscalar(x));
            parse(p, varargin{:});

            % Hyperparam
            convSpecs      = p.Results.CNN;
            rnnSpecs       = p.Results.RNN;
            fcSpecs        = p.Results.FC;
            obj.tPool      = p.Results.tPool;
            obj.numClasses = p.Results.numClasses;
            obj.timeStep   = p.Results.timeStep;

            % Build convolutional and pooling layers
            obj.CNNmodule = cell(size(convSpecs));
            for i = 1:numel(convSpecs)
                spec = convSpecs{i};
                switch spec{1}
                    case 'conv'
                        obj.CNNmodule{i} = ConvolutionalLayer(spec{2},spec{3},spec{4},spec{5});
                    case 'pool'
                        obj.CNNmodule{i} = PoolingLayer(spec{2});
                    otherwise
                        error('Unknown conv spec: %s', spec{1});
                end
            end
        
            % Build recurrent layers
            obj.RNNmodule = cell(size(rnnSpecs));
            for i = 1:numel(rnnSpecs)
                spec = rnnSpecs{i};
                switch spec{1}
                    case 'rnn'
                        obj.RNNmodule{i} = RecurrentUnit(spec{2},spec{3},spec{4});
                    case 'lstm'
                        obj.RNNmodule{i} = LSTMUnit(spec{2},spec{3});
                    case 'gru'
                        obj.RNNmodule{i} = GRUnit(spec{2},spec{3});
                    otherwise
                        error('Unknown RNN spec: %s', spec{1});
                end
            end
        
            % Build fully connected classifier
            obj.FCmodule = cell(size(fcSpecs));
            if ~isempty(fcSpecs)
                if numel(fcSpecs{1}) < 2
                    error('fcSpecs must contain at least two layer sizes (input and output).');
                else
                    obj.FCmodule = FullyConnectedNetwork(fcSpecs);
                end
            end
        
            % Derived window parameters
            % windowSize: number of raw timesteps covered by one RNN input frame (multiple timesteps pooled together)
            % stepSize: downsampled step between successive RNN inputs (single timestep)
            obj.CNNwindowSize = 1;
            obj.CNNstepSize   = 1;
            for i = numel(obj.CNNmodule):-1:1
                layer = obj.CNNmodule{i};
                if isa(layer, 'ConvolutionalLayer')
                    obj.CNNwindowSize = obj.CNNwindowSize + obj.CNNmodule{i}.kernelSize - 1;
                elseif isa(layer, 'PoolingLayer')
                    obj.CNNwindowSize = obj.CNNwindowSize * obj.CNNmodule{i}.poolingRatio;
                    obj.CNNstepSize   = obj.CNNstepSize*obj.CNNmodule{i}.poolingRatio;
                end
            end
        
            % Learning parameters
            obj.eta               = p.Results.eta;
            obj.learningRateDecay = p.Results.learningRateDecay;
            obj.beta_1            = p.Results.beta_1;
            obj.beta_2            = p.Results.beta_2;
            obj.epsilon           = 10^-8;  % Small number to avoid division-by-zero
            obj.t                 = 1;      % Adam time-step counter
        
            % Initialize learning metrics and history
            initialAcc       = evaluate(obj, testData);
            % learningHistory columns: [batchSize, avgSegmLen, eta, valAccuracy, trainAccuracy, loss, elapsedTime]
            obj.learningHistory      = [0, 0, 0, initialAcc, NaN, NaN, NaN];
            obj.totalEntropy_vec     = [];
            obj.trainingaccuracy_vec = [];
        end
    
        %% Forward: Execute the network on an input sequence
        function networkOutput = forward(obj, inputSequence)
            % Input:  inputSequence - [inputLength × numChannels] raw time-series
            % Output: networkOutput - [numSteps × numClasses] softmax probabilities per time step
    
            % CNN
            for i = 1:numel(obj.CNNmodule)
                inputSequence = obj.CNNmodule{i}.forward(inputSequence);
            end
        
            % Number of RNN/FC steps
            if isempty(obj.RNNmodule)
                numSteps = floor((size(inputSequence,1)-obj.tPool+1)/obj.tPool);
            else
                numSteps = floor(size(inputSequence,1)/obj.tPool);
            end

            networkOutput = zeros(numSteps,obj.numClasses);
        
            % Process each time step by RNN and FCL
            for j = 1:numSteps
                % Extract the j-th block of size [numTimeSegments × featureDim]
                startIdx = (j-1)*obj.tPool+1;
                block = inputSequence(startIdx:startIdx+obj.tPool-1,:);
                
                % Flatten into a row vector for RNN/FC input
                inputStep = reshape(block,1,[]);
        
                % RNN
                for i = 1:numel(obj.RNNmodule)
                    inputStep = obj.RNNmodule{i}.forward(inputStep);
                end
        
                % FCL
                if ~isempty(obj.FCmodule)
                    inputStep = obj.FCmodule.forward(inputStep);
                end
        
                % Softmax
                networkOutput(j,:) = softmax(inputStep);
            end
        end
    
        %% Train: Optimize network
        function train(obj, trainingData, testData, epochs, batchSize, varargin)
            % Inputs:
            %   trainingData : cell array {sequence, labels}
            %   testData     : data for validation in each epoch
            %   epochs       : number of full data passes
            %   numSegments  : number of non-overlapping segments
            %   batchSize    : segments per update
            %   trains with no regard to temporal dependencies if no numSegments is given (e.g. in case of no RNN module)

            p = inputParser;
            addParameter(p,'numSegments', [], @(x) isnumeric(x));
            parse(p, varargin{:});
            numSegments = p.Results.numSegments;

            % Compute label frequencies for weighted loss
            labelWeights = countTrainingLabels(trainingData, obj.numClasses);

            % Precompute segment info: [trialIdx, startIdx, endIdx]
            segmInfos = segmentSequences(obj, trainingData, numSegments);

            numConvLayers = numel(obj.CNNmodule);
            numRecLayers  = numel(obj.RNNmodule);

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
                    
                    % Go through the segments
                    for segmIdx = 1:size(batchInfo,1)
        
                        % reset hiddenstates and stored activations in the RNN and FC layers
                        obj.resetMemory();
        
                        tr       = batchInfo(segmIdx,1);
                        startIdx = batchInfo(segmIdx,2);
                        endIdx   = batchInfo(segmIdx,3);
        
                        trainingSegment = trainingData{tr,1}(startIdx:endIdx,:);
                        labels          = trainingData{tr,2}(startIdx:endIdx,:);
        
                        % CNN forward pass
                        for i = 1:numConvLayers
                            trainingSegment = obj.CNNmodule{i}.forward(trainingSegment, 'train');
                        end
        
                        outputLength = floor(size(trainingSegment,1)/obj.tPool);
                        output       = zeros(outputLength, obj.numClasses);
        
                        % Forward pass though RNN and FC
                        for scanningIdx = 1:outputLength
                            startIdx   = (scanningIdx-1)*obj.tPool+1;
                            endIdx     = startIdx + obj.tPool-1;
                            inputSlice = reshape(trainingSegment(startIdx:endIdx,:,:), 1, []);
        
                            % RNN
                            for i = 1:numRecLayers
                                inputSlice = obj.RNNmodule{i}.forward(inputSlice, 'train');
                            end
                            
                            % FC
                            if ~isempty(obj.FCmodule)
                                inputSlice = obj.FCmodule.forward(inputSlice, 'train');
                            end
        
                            % Softmax
                            output(scanningIdx,:) = softmax(inputSlice);
                        end
        
                        % Storage for backpropagated error at the CNN output
                        if ~isempty(obj.CNNmodule)
                            convIdx = find(cellfun(@(L) isa(L,'ConvolutionalLayer'), obj.CNNmodule), 1, 'last');
                            mapSz   = size(obj.CNNmodule{convIdx}.preacts);  % [T_down x featDim1 x featDim2]
                            dxCNN   = zeros([outputLength*obj.tPool, mapSz(2:end)]);
                        end
                            
                        % Backward pass from softmax through FC, RNN, CNN
                        for scanningIdx = outputLength:-1:1
                            % Compute average ground-truth over window
                            startIdx = (scanningIdx-1)*(obj.CNNstepSize*obj.tPool)+1;
                            endIdx   = min(startIdx+obj.CNNstepSize*(obj.tPool-1)+obj.CNNwindowSize-1,size(labels,1));
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
                            if ~isempty(obj.FCmodule)
                                dx = obj.FCmodule.backprop(dx, scanningIdx);
                            end

                            % Backprop through RNN
                            for i = numRecLayers:-1:1
                                dx = obj.RNNmodule{i}.backprop(dx, scanningIdx);
                            end
        
                            % Reshape and accumulate CNN error      
                            if ~isempty(obj.CNNmodule)
                                dx     = reshape(dx, [obj.tPool, mapSz(2:end)]);
                                startT = (scanningIdx-1)*obj.tPool + 1;
                                endT   = startT + obj.tPool - 1;
                                dxCNN(startT:endT, :, :) = dx;
                            end
                        end
        
                        % Backprop through CNN
                        for i = numConvLayers:-1:1
                            dxCNN = obj.CNNmodule{i}.backprop(dxCNN);
                        end
        
                        totalSamples_batch = totalSamples_batch + outputLength;
                    end
        
                    % Adam parameter update
                    adamOptimizer(obj, totalSamples_batch);
                    obj.t = obj.t + 1;
        
                    % Record batch metrics
                    obj.totalEntropy_vec     = [obj.totalEntropy_vec;     totalEntropy_batch/totalSamples_batch];
                    obj.trainingaccuracy_vec = [obj.trainingaccuracy_vec; correctCount_batch/totalSamples_batch];
        
                    % Plot progress
                    subplot(2,1,1); cla;
                    plot(obj.totalEntropy_vec);
                    grid on; %axis padded;
                    title('Training loss (avg cross-entropy)');
                    xlabel('Batch'); ylabel('Loss');

                    subplot(2,1,2); cla;
                    plot(obj.trainingaccuracy_vec);
                    grid on; %axis padded;
                    title('Training accuracy');
                    xlabel('Batch'); ylabel('Accuracy');

                    drawnow
        
                    % Update epoch totals
                    correctCount_epoch = correctCount_epoch + correctCount_batch;
                    totalEntropy_epoch = totalEntropy_epoch + totalEntropy_batch;
                    totalSamples_epoch = totalSamples_epoch + totalSamples_batch;
                end
        
                elapsedTime = toc;
                fprintf('Epoch %d complete\n', epochIdx);
        
                % Update learningHistory log
                trainingAccuracy = correctCount_epoch/totalSamples_epoch;
                residual         = totalEntropy_epoch/totalSamples_epoch;
                testAcc          = evaluate(obj, testData);
                avgSegmLen       = totalSamples_epoch/size(segmInfos,1);
                obj.learningHistory(end+1,:) = [batchSize, avgSegmLen, obj.eta, testAcc, trainingAccuracy, residual, elapsedTime];
                
                % Learning rate decay
                obj.eta = obj.learningRateDecay*obj.eta;
        
                % Save trained network
                timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
                filename  = sprintf('Network_trained_%s.mat', timestamp);
                save(filename, 'obj');
        
            end
        end

        %% Evaluate: Compute overall classification accuracy on test set
        function acc = evaluate(obj, testData)
            % Inputs: 
            %   obj         - TemporalNeuralNet instance
            %   testData    - N×2 cell array, each row: {rawSequence, oneHotLabels}
            % Output:
            %   acc         - fraction of correctly predicted time‐segments

            % Initialize counters
            correctCount = 0;
            totalFrames  = 0;
        
            % Loop through each item in the test data
            for i = 1:size(testData, 1)
                input  = testData{i,1};    % [T_raw × numChannels]
                labels = testData{i,2};    % [T_raw × numClasses]
                obj.resetMemory();         % clear any RNN/FC hidden state
        
                % Forward-pass through full network
                outputActivations = forward(obj, input);
        
                % Determine the predicted class
                [~, predictedClasses] = max(outputActivations,[],2);
                outputLength          = numel(predictedClasses);
                
                % For each segment, derive the “true” class at the segment midpoint
                for j = 1:outputLength
                    % Compute raw‐data window corresponding to segment t
                    startIdx = (j-1)*(obj.CNNstepSize*obj.tPool)+1;
                    endIdx   = min(startIdx+obj.CNNstepSize*(obj.tPool-1)+obj.CNNwindowSize-1,size(labels,1));
                    midpoint = floor((startIdx + endIdx)/2);

                    % True class at midpoint
                    [~, trueClass] = max(labels(midpoint,:), [], 2);

                    % Compare prediction with truth
                    if predictedClasses(j) == trueClass
                        correctCount = correctCount + 1;
                    end
                end
        
                % Accumulate count of evaluated segments
                totalFrames = totalFrames + outputLength;
                
            end
        
            % Overall success rate
            acc = correctCount/totalFrames;
        
        end

        %% adamOptimizer: caller function of 'applyAdam' in each layer 
        function adamOptimizer(obj, totalSamples_batch)
            % ADAMOPTIMIZER: Adam updates across all modules
            % totalSamples_batch: normalizing factor (batch size)

            for L = obj.CNNmodule
                L{1}.applyAdam(obj.eta, obj.beta_1, obj.beta_2, totalSamples_batch, obj.t, obj.epsilon);
            end
            for U = obj.RNNmodule
                U{1}.applyAdam(obj.eta, obj.beta_1, obj.beta_2, totalSamples_batch, obj.t, obj.epsilon);
            end
            if ~isempty(obj.FCmodule)
                obj.FCmodule.applyAdam(obj.eta, obj.beta_1, obj.beta_2, totalSamples_batch, obj.t, obj.epsilon);
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
                windowSize = (obj.CNNwindowSize+(obj.tPool-1)*obj.CNNstepSize);
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
            for i = 1:numel(obj.CNNmodule)
                obj.CNNmodule{i}.resetStoredActivations();
            end
            % Reset each RNN unit's hidden state history
            for i = 1:numel(obj.RNNmodule)
                obj.RNNmodule{i}.resetMemory();
            end
            % Reset FC module if present (clears any stored activations)
            if ~isempty(obj.FCmodule)
                obj.FCmodule.resetStoredActivations();
            end
        end

        %% resetGrads: zero out gradient accumulators before each batch
        function resetGrads(obj)
            % Reset CNN gradient buffers
            for i = 1:numel(obj.CNNmodule)/2 % build check fot layer type
                obj.CNNmodule{(i-1)*2+1}.resetGrads();
            end
            % Reset RNN gradient accumulators
            for i = 1:numel(obj.RNNmodule)
                obj.RNNmodule{i}.resetGrads();
            end
            % Reset FC gradients if module exists
            if ~isempty(obj.FCmodule)
                obj.FCmodule.resetGrads();
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