classdef FullyConnectedNetwork < handle
    %% FullyConnectedNetwork: sequence of dense layers with leaky-ReLU activations
    % Supports forward inference, backprop per time-step for truncated BPTT, and Adam updates for weights/biases.
    properties
        numLayers
        sizes
        numParams
        biases
        weights
        actCache
        preactCache
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
            obj.numLayers   = numel(sizes)-1;
            obj.sizes       = sizes;

            % Allocate storage for parameters between layers
            obj.biases      = cell(obj.numLayers,1);
            obj.weights     = cell(obj.numLayers,1);

            % Caches for activations per time-step
            obj.actCache    = cell(obj.numLayers+1,1);
            obj.preactCache = cell(obj.numLayers+1,1);

            % Leaky ReLU parameter
            obj.a           = 0.01;
            
            % He initialization for weights and biases
            for i = 1:obj.numLayers
                obj.weights{i} = randn(obj.sizes(i+1), obj.sizes(i))*sqrt(2/obj.sizes(i));
                obj.biases{i}  = -mean(obj.weights{i},2);
            end

            % Number of parameters per layer
            obj.numParams = 0;
            for i = 1:obj.numLayers
                numLayerParam = numel(obj.weights{i}) + numel(obj.biases{i});
                obj.numParams = obj.numParams + numLayerParam;
            end

            % Initialize optimizer and gradient buffers
            for j = 1:numel(obj.weights)
                obj.vdw{j,1} = zeros(size(obj.weights{j}));
                obj.sdw{j,1} = zeros(size(obj.weights{j}));
                obj.vdb{j,1} = zeros(size(obj.biases{j}));
                obj.sdb{j,1} = zeros(size(obj.biases{j}));

                obj.dW{j,1}  = zeros(size(obj.weights{j}));
                obj.db{j,1}  = zeros(size(obj.biases{j}));
            end
        end

        %% Forward: process one time-step through all dense layers
        function a_out = forward(obj, x_in, numSteps, t, varargin)
            % Inputs:
            %   x_in     - input data, row vector  [1×inputSize]
            %   numSteps - length of the training sequence
            %   t        - current timestep of the training sequence
            % Output:
            %   a_out    - output activations, row vector [1×outputSize]

            doCache = ~isempty(varargin) && strcmp(varargin{1}, 'train');

            % Column vector for matrix operations
            a_out   = x_in';  

            % Optionally cache input as layer 1 activations
            if doCache
                if isempty(obj.actCache{1})
                    for i = 1:obj.numLayers+1
                        obj.preactCache{i} = zeros(numSteps,obj.sizes(i));
                        obj.actCache{i}    = zeros(numSteps,obj.sizes(i));
                    end
                end
                obj.preactCache{1}(t,:) = x_in;
                obj.actCache{1}(t,:)    = x_in;
            end

            % Propagate through each layer
            for i = 1:obj.numLayers
                z = obj.weights{i} * a_out + obj.biases{i};
                
                if i < obj.numLayers
                    a_out = leakyReLU(z, obj.a);
                else
                    a_out = z;
                end

                if doCache
                    obj.preactCache{i+1}(t,:) = z';
                    obj.actCache{i+1}(t,:)    = a_out';
                end
            end

            % Return as row vector [1×outputDim]
            a_out = a_out';
        end

        %% Backprop: compute gradients for time-step t
        function [d_in, wUpdate] = backprop(obj, d_out, t, idxMap)
            % Inputs:
            %   d_out   - gradient from softmax layer, [1×outputDim]
            %   t       - time index corresponding to cached activations
            %   idxMap  - index map with separator indices of the weight update vector
            % Outputs:
            %   d_in    - backpropagated error at the input
            %   wUpdate - update vector of weights and biases from timestep t

            % Initialize local gradient storage
            wUpdate = zeros(obj.numParams, 1);

            % Column vector for matrix operations
            d_out = d_out';
        
            % Iterate backwards through layers
            for i = obj.numLayers:-1:1
                % Retrieve preactivation in layer i at time t_idx
                z = obj.preactCache{i+1}(t,:);
                
                if i < obj.numLayers
                    delta = d_out .* leakyReLU_prime(z, obj.a);
                else
                    delta = d_out;
                end

                % Bias gradient
                startBias = idxMap.fcEnd(i) - numel(obj.biases{i}) + 1;
                endBias   = idxMap.fcEnd(i);
                wUpdate(startBias:endBias) = delta;

                % Weight gradient: dL/dW = delta * a_prev^T
                startWeights = idxMap.fcStrt(i);
                endWeights   = idxMap.fcEnd(i) - numel(obj.biases{i});
                wUpdate(startWeights:endWeights) = delta * obj.actCache{i}(t,:);

                % Error to previous layer
                d_out = obj.weights{i}' * delta;
            end

            % Back to row for CNN/RNN input
            d_in = d_out';
        end

        %% ApplyAdam: update weights and biases using Adam formula
        function applyAdam(obj, eta, beta1, beta2, m, t, eps)
            [obj.weights, obj.vdw, obj.sdw] = adamUpdate(obj.weights, obj.dW, obj.vdw, obj.sdw, beta1, beta2, t, eta, m, eps);
            [obj.biases,  obj.vdb, obj.sdb] = adamUpdate(obj.biases,  obj.db, obj.vdb, obj.sdb, beta1, beta2, t, eta, m, eps);
        end

        %% Resets
        function resetStoredActivations(obj)
            % Clear cached activations for a new sequence
            obj.actCache    = cell(obj.numLayers+1,1);
            obj.preactCache = cell(obj.numLayers+1,1);
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
function s = leakyReLU(z, a)
    s = max(a*z,z);
end

function s = leakyReLU_prime(z, a)
    s = ones(length(z),1);
    s(z < 0) = a;
end