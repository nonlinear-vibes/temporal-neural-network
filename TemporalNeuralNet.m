classdef (Abstract) TemporalNeuralNet < handle
    %% TEMPORALNEURALNET (abstract): shared CNN -> RNN/GRU/LSTM -> FC backbone
    %
    % Abstract base class for the identical aspects of Classifier and Predictor:
    % layer construction, the idxMap gradient-slicing index, the shared
    % CNN->RNN->FC forward pass for a single fixed-length window (forward), the
    % Adam optimizer wiring, and the memory/gradient reset helpers.
    %
    % ARCHITECTURE
    %  Input sequence -> [CNN/pooling layers] -> [RNN/GRU/LSTM layers] -> 
    %  -> [fully connected head] -> per-step output
    %  All three stages are optional, but the order is fixed. 
    % 
    %  Shared:   constructor (layer/idxMap/optimizer setup), forward(),
    %            adamOptimizer(), resetMemory(), resetGrads()
    %  Abstract: train(), backwardPass(), evaluate(), segmentSequences(),
    %            outputActivation()
    %
    %  Classifier: dense per-timestep classification, cross-entropy loss,
    %              windowed-label targets
    %  Predictor:  autoregressive multi-step forecast, MSE loss, 
    %              next-raw-value targets


    properties
        cnnModule             % cell array of ConvolutionalLayer/PoolingLayer objects, in order (may be empty)
        rnnModule             % cell array of RecurrentUnit/GRUnit/LSTMUnit objects, in order (may be empty)
        fcModule              % single FullyConnectedNetwork object (may be empty)
        tPool                 % number of consecutive CNN-output steps grouped into one RNN/FC input frame
        outputWidth           % final per-step output width; set by the subclass constructor
        numChannels           % number of raw input channels (columns of the input sequence)
        timeStep              % stride between evaluated steps when no RNN module is present (ignored otherwise, where tPool is the stride instead)
        cnnWindowSize         % raw timesteps covered by one CNN output step (receptive field width)
        cnnStepSize           % raw stride of the CNN window (1 with no pooling)
        idxMap                % index boundaries used to slice flattened accumulated gradient vectors back into per-layer weight/bias tensors
        eta                   % Adam learning rate (decays each epoch via learningRateDecay)
        beta_1                % Adam beta1 (first-moment decay)
        beta_2                % Adam beta2 (second-moment decay)
        epsilon               % small constant added for numerical stability (Adam denominator, log(), etc.)
        learningRateDecay     % multiplicative decay applied to eta after each epoch
        t                     % Adam timestep counter (increments once per batch, drives bias correction)
        learningHistory       % per-epoch log; column layout is subclass-specific
        totalLossHistory      % per-batch loss (cross-entropy for Classifier, MSE for Predictor)
        trainingMetricHistory % per-batch secondary metric (accuracy for Classifier, MAE for Predictor)
    end
	
    methods
        %% TemporalNeuralNet constructor: shared layer/idxMap/optimizer setup
        % Not called directly (Abstract class), invoked via obj@TemporalNeuralNet(...) 
        % from a subclass constructor.
        %
        % Inputs (all name-value, all optional):
        %   'CNN'   - cell array of layer specs ({}, i.e. no CNN, by default):
        %               {'conv', numChannels, inFeatures, outFeatures, kernelSize}
        %               {'pool', poolingRatio}
        %             inFeatures of each conv layer must equal outFeatures of the previous
        %             one (or 1, for the first layer).
        %   'RNN'   - cell array of recurrent layer specs ({} by default), each:
        %               {'rnn',  inDim, hiddenDim, outDim}
        %               {'gru',  inDim, outDim}
        %               {'lstm', inDim, outDim}
        %             inDim of the first layer must equal the CNN module's output width * tPool
        %             (or numChannels * tPool if there's no CNN module); inDim of later layers must
        %             equal the previous layer's outDim.
        %   'FC'    - fully connected head spec: { [in, h1, ..., out] }, a single cell containing 
        %             one size vector. Constructor errors if given but with fewer than 2 sizes; 
        %             an entirely empty {} leaves fcModule as an empty
        %             cell.
        %   'tPool' - number of consecutive raw/CNN-output steps grouped into one RNN/FC input
        %             frame (default 1)
        %   'timeStep' - stride between evaluated steps when there's no RNN module, used to downsample
        %             the input when there's not much new information between the input windows 
        %             (default: tPool).
        %             Ignored with a warning, if an RNN module is present (tPool is always the stride
        %             in that case, since the RNN's hidden state needs every step to be seen in order).
        %   'numChannels' - number of raw input channels (default 1)
        %   'eta'   - initial Adam learning rate (default 0.5)
        %   'learningRateDecay' - multiplicative decay applied to eta after every epoch (default 0.95)
        %   'beta_1', 'beta_2' - Adam moment-decay hyperparameters (defaults 0.90, 0.999)
        %
        % Output: obj, with cnnModule/rnnModule/fcModule/idxMap built and optimizer state initialized.
        % 


        function obj = TemporalNeuralNet(varargin)
            p = inputParser;
            addParameter(p,'CNN',               {},   @(x) iscell(x));
            addParameter(p,'RNN',               {},   @(x) iscell(x));
            addParameter(p,'FC',                {},   @(x) iscell(x));
            addParameter(p,'tPool',             1,    @(x) isnumeric(x)&&isscalar(x));
            addParameter(p,'eta',               0.5,  @(x) isnumeric(x)&&isscalar(x));
            addParameter(p,'learningRateDecay', 0.95, @(x) isnumeric(x)&&isscalar(x));
            addParameter(p,'beta_1',            0.90, @(x) isnumeric(x)&&isscalar(x));
            addParameter(p,'beta_2',            0.999,@(x) isnumeric(x)&&isscalar(x));
            addParameter(p,'timeStep',          [],   @(x) isnumeric(x)&&isscalar(x));
            addParameter(p,'numChannels',       1,    @(x) isnumeric(x)&&isscalar(x));
            parse(p, varargin{:});
 
            convSpecs       = p.Results.CNN;
            rnnSpecs        = p.Results.RNN;
            fcSpecs         = p.Results.FC;
            obj.tPool       = p.Results.tPool;
            obj.numChannels = p.Results.numChannels;
 
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
 
            % Build fully connected head
            obj.fcModule = cell(size(fcSpecs{1}));
            if ~isempty(fcSpecs)
                if numel(fcSpecs{1}) < 2
                    error('fcSpecs must contain at least two sizes (input and output).');
                else
                    obj.fcModule = FullyConnectedNetwork(fcSpecs{1});
                end
            end
 
            % Derived window parameters
            obj.cnnWindowSize = 1;
            obj.cnnStepSize   = 1;
            for i = numel(obj.cnnModule):-1:1
                layer = obj.cnnModule{i};
                if isa(layer, 'ConvolutionalLayer')
                    obj.cnnWindowSize = obj.cnnWindowSize + layer.kernelSize - 1;
                elseif isa(layer, 'PoolingLayer')
                    obj.cnnWindowSize = obj.cnnWindowSize * layer.poolingRatio;
                    obj.cnnStepSize   = obj.cnnStepSize   * layer.poolingRatio;
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
 
            obj.idxMap.rnn = cell(numel(obj.rnnModule),1);
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
                        strtIdx   = strtIdx + numParams;
                    end
                end
            else
                obj.idxMap.rnn{1} = [0, 0];
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
            obj.epsilon           = 1e-8;
            obj.t                 = 1;

            obj.totalLossHistory      = [];
            obj.trainingMetricHistory = [];
        end
 
        %% forward: shared CNN -> RNN -> FC pass over one fixed window
        % Runs ONE pass over inputSequence and returns one output row per (tPool-sized)
        % step within it, not autoregressive, not aware of any training loss. Classifier
        % uses this directly as its own public forward(); Predictor wraps it in forecast()/
        % forecastSequence() for multi-step rollout The only behavioral difference between
        % subclasses is the final activation applied to each step's raw output (softmax vs.
        % identity).
        %
        % Inputs:
        %   inputSequence - [T x numChannels] raw window. T must be at least tPool. How many
        %                   output rows you get back depends on tPool/timeStep.
        %   'isTraining'  - (name-value pair, default false) if true, each layer caches its
        %                   activations for backpropagation (obj.backwardPass() reads these
        %                   afterward). If false, layers only keep enough state for a single
        %                   forward pass.
        %
        % Output:
        %   stepOutput - [numSteps x outputWidth], where numSteps = floor((T-tPool)/stepSize)+1
        %                and stepSize is tPool if an RNN module is present, or timeStep 
        %                otherwise.
        function stepOutput = forward(obj, inputSequence, varargin)
            p = inputParser;
            addParameter(p,'isTraining', false, @(x) islogical(x));
            parse(p, varargin{:});
            isTraining = p.Results.isTraining;
 
            for i = 1:numel(obj.cnnModule)
                inputSequence = obj.cnnModule{i}.forward(inputSequence, isTraining);
            end
 
            if isempty(obj.rnnModule)
                stepSize = obj.timeStep;
            else
                stepSize = obj.tPool;
            end
            numSteps   = floor((size(inputSequence,1)-obj.tPool)/stepSize) + 1;
            stepOutput = zeros(numSteps, obj.outputWidth);
 
            for j = 1:numSteps
                startIdx = (j-1)*stepSize+1;
                block = inputSequence(startIdx:startIdx+obj.tPool-1,:);
                inputStep = reshape(block,1,[]);
                for i = 1:numel(obj.rnnModule)
                    inputStep = obj.rnnModule{i}.forward(inputStep, numSteps, j, 'train', isTraining);
                end
                if ~isempty(obj.fcModule)
                    inputStep = obj.fcModule.forward(inputStep, numSteps, j, 'train', isTraining);
                end
                stepOutput(j,:) = obj.outputActivation(inputStep);
            end
        end
 
        %% adamOptimizer: caller function of 'applyAdam' in each layer
        % Input: totalSamples_batch - normalizing factor for the step size
        % 
        % Produces no output, updates the weights and biases as a side effect
        function adamOptimizer(obj, totalSamples_batch)
            for i = 1:numel(obj.cnnModule)
                if isa(obj.cnnModule{i}, "ConvolutionalLayer")
                    obj.cnnModule{i}.applyAdam(obj.eta, obj.beta_1, obj.beta_2, totalSamples_batch, obj.t, obj.epsilon);
                end
            end
            for i = 1:numel(obj.rnnModule)
                obj.rnnModule{i}.applyAdam(obj.eta, obj.beta_1, obj.beta_2, totalSamples_batch, obj.t, obj.epsilon);
            end
            if ~isempty(obj.fcModule)
                obj.fcModule.applyAdam(obj.eta, obj.beta_1, obj.beta_2, totalSamples_batch, obj.t, obj.epsilon);
            end
        end

        %% resetMemory: clear stored activations and hidden states before each new sequence
        function resetMemory(obj)
            for i = 1:numel(obj.cnnModule)
                obj.cnnModule{i}.resetStoredActivations();
            end
            for i = 1:numel(obj.rnnModule)
                obj.rnnModule{i}.resetMemory();
            end
            if ~isempty(obj.fcModule)
                obj.fcModule.resetStoredActivations();
            end
        end
 
        %% resetGrads: zero out gradient accumulators before each batch
        function resetGrads(obj)
            for i = 1:numel(obj.cnnModule)
                if isa(obj.cnnModule{i}, "ConvolutionalLayer")
                    obj.cnnModule{i}.resetGrads();
                end
            end
            for i = 1:numel(obj.rnnModule)
                obj.rnnModule{i}.resetGrads();
            end
            if ~isempty(obj.fcModule)
                obj.fcModule.resetGrads();
            end
        end
    end
 
    methods (Abstract)
        % Full training loop: training data segmentation (via segmentSequences),
        % batching, backward pass (via backwardPass), Adam updates, per-epoch
        % validation reporting (via evaluate), and learningHistory/checkpoint
        % bookkeeping.
        train(obj, trainingData, validationData, epochs, batchSize, varargin)

        % Backprop through FC/RNN/CNN for one sequence's worth of cached activations
        % (i.e. one forward() call made with 'isTraining',true). Loss function and
        % target lookup differ between subclasses (cross-entropy vs. MSE;
        % windowed-label-average vs. next-raw-value).
        [cnnSeqUpdate, rnnSeqUpdate, fcSeqUpdate, totalLoss, totalSamples] = backwardPass(obj, output, target)

        % Metric computation over a dataset (accuracy for Classifier, MAE for Predictor).
        % Signature differs completely per subclass: Predictor additionally needs
        % forecastLength/segmentLength to know how far ahead and from how much context
        % to forecast.
        metricOut = evaluate(obj, data, varargin)

        % Windowing strategy used by train() to slice each trial's raw% sequence into
        % training segments. Differs completely between subclasses:
        % Classifier: overlapping fixed-count segments;
        % Predictor: one-step-shifted windows with reserved trailing target steps
        segmInfos = segmentSequences(obj, data, varargin)

        % Final per-timestep activation applied inside forward(), on each step's raw
        % FC output z.
        % Classifier: softmax(z)
        % Predictor: z unchanged
        a = outputActivation(obj, z)
    end
end