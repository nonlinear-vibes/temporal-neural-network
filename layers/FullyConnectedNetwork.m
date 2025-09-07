classdef FullyConnectedNetwork < handle
    %% FullyConnectedNetwork: sequence of dense layers with leaky-ReLU activations
    % Supports forward inference, backprop per time-step for truncated BPTT, and Adam updates for weights/biases.
    properties
        numLayers
        sizes
        biases
        weights
        acts
        preacts
        dW
        db
        vdw
        vdb
        sdw
        sdb
        a

    end
    
    %%
    methods
        %% FullyConnectedNetwork constructor: initialize weights, biases, and optimizer buffers
        function obj = FullyConnectedNetwork(sizes)
            % Constructor
            obj.numLayers = length(sizes{1})-1;
            obj.sizes     = sizes{1};

            % allocate storage for parameters between layers
            obj.biases    = cell(obj.numLayers,1);
            obj.weights   = cell(obj.numLayers,1);

            % caches for activations per time-step
            obj.acts      = cell(obj.numLayers+1,1);
            obj.preacts   = cell(obj.numLayers+1,1);

            % Leaky ReLU parameter
            obj.a = 0.01;
            
            % He initialization for weights and biases
            for i = 1:obj.numLayers
                obj.weights{i} = randn(obj.sizes(i+1), obj.sizes(i))*sqrt(2/obj.sizes(i));
                obj.biases{i}  = -mean(obj.weights{i},2);
            end

            % Initialize optimizer and gradient buffers
            for j = 1:numel(obj.weights)
                obj.vdw{j,1} = zeros(size(obj.weights{j}));
                obj.sdw{j,1} = zeros(size(obj.weights{j}));
                obj.vdb{j,1} = zeros(size(obj.biases{j}));
                obj.sdb{j,1} = zeros(size(obj.biases{j}));

                obj.dW{j,1} = zeros(size(obj.weights{j}));
                obj.db{j,1} = zeros(size(obj.biases{j}));
            end
        end

        %% Forward: process one time-step through all dense layers
        function as = forward(obj, input, varargin)
            % input - [1×inputDim]

            doCache = ~isempty(varargin) && strcmp(varargin{1}, 'train');

            % column vector for matrix operations
            as = input';  

            % optionally cache input as layer 1 activations
            if doCache
                obj.preacts{1}(end+1,:) = input;
                obj.acts{1}(end+1,:)    = input;
            end
            
            % propagate through each layer
            for i = 1:obj.numLayers
                z  = obj.weights{i} * as + obj.biases{i};
                as = ReLU(z, obj.a);

                if doCache
                    obj.preacts{i+1} = [obj.preacts{i+1}; z'];
                    obj.acts{i+1}    = [obj.acts{i+1};    as'];
                end
            end

            % return as row vector [1×outputDim]
            as = as';
        end

        %% Backprop: compute gradients for time-step t
        function d_in = backprop(obj, d_out, t)
            % d_out - gradient from softmax layer, [1×outputDim]
            % t: time index corresponding to cached activations

            % initialize local gradient storage
            dW_new = cell(size(obj.weights));
            db_new = cell(size(obj.biases));

            % column vector for matrix operations
            d_out = d_out';
        
            % iterate backwards through layers
            for i = obj.numLayers:-1:1
                % retrieve preactivation in layer i at time t
                z         = obj.preacts{i+1}(t,:);
                ReLu_p    = ReLU_prime(z, obj.a);

                delta     = d_out .* ReLu_p;
                % bias gradient
                db_new{i} = delta;
                % weight gradient: delta * activation(i)
                dW_new{i} = delta * obj.acts{i}(t,:);
                % error to previous layer
                d_out      = obj.weights{i}' * delta;
            end

            % back to row for CNN/RNN input
            d_in = d_out';

            % accumulate gradients
            obj.dW = cellfun(@(x,y) x + y, obj.dW, dW_new, 'UniformOutput', false);
            obj.db = cellfun(@(x,y) x + y, obj.db, db_new, 'UniformOutput', false);

        end

        %% ApplyAdam: update weights and biases using Adam formula
        function applyAdam(obj, eta, beta1, beta2, m, t, eps)
            [obj.weights, obj.vdw, obj.sdw] = adamUpdate(obj.weights, obj.dW, obj.vdw, obj.sdw, beta1, beta2, t, eta, m, eps);
            [obj.biases,  obj.vdb, obj.sdb] = adamUpdate(obj.biases,  obj.db, obj.vdb, obj.sdb, beta1, beta2, t, eta, m, eps);
        end

        %% Resets
        function resetStoredActivations(obj)
            % Clear cached activations for a new sequence
            obj.acts    = cell(obj.numLayers+1,1);
            obj.preacts = cell(obj.numLayers+1,1);
        end

        function resetGrads(obj)
            % Zero gradient accumulators before each batch
            for i = 1:obj.numLayers
                obj.dW{i} = zeros(size(obj.weights{i}));
                obj.db{i} = zeros(size(obj.biases{i}));
            end
        end
    end
end


%%
function s = ReLU(z, a)
    s = max(a*z,z);
end

function s = ReLU_prime(z, a)
    s = ones(length(z),1);
    s(z < 0) = a;
end