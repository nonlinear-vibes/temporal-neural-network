classdef (Abstract) TemporalNeuralNet < handle
    %% TEMPORALNEURALNET (abstract): shared CNN -> RNN/GRU/LSTM -> FC backbone
    %
    % Abstract base class for the identical aspects of Classifier and Predictor: 
    % layer construction, the idxMap gradient-slicing index, the shared
    % CNN->RNN->FC forward pass for a single fixed-length window (forwardStep),
    % the Adam optimizer wiring, and the memory/gradient reset helpers.
 
    properties
        cnnModule
        rnnModule
        fcModule
 
        tPool
        outputWidth
        numChannels
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
 
        learningHistory       % per-epoch log
        totalLossHistory      % per-batch loss (cross-entropy for Classifier, MSE for Predictor
        trainingMetricHistory % per-batch metric (accuracy for Classifier, MAE for Predictor)
    end
 
    methods
        %% TemporalNeuralNet constructor: shared layer/idxMap/optimizer setup
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
            addParameter(p,'numChannels',       16,   @(x) isnumeric(x)&&isscalar(x));
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
 
        %% forwardStep: shared CNN -> RNN -> FC pass over one fixed window
        % The only difference between Classifier and Predictor is the final
        % activation (softmax vs identity).
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
        train(obj, trainingData, validationData, epochs, batchSize, varargin)
        [cnnSeqUpdate, rnnSeqUpdate, fcSeqUpdate, totalLoss, totalSamples] = backwardPass(obj, output, target)
        metricOut = evaluate(obj, data, varargin)
        segmInfos = segmentSequences(obj, data, varargin)
        a = outputActivation(obj, z)
    end
end
