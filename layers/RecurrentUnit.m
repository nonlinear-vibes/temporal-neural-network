classdef RecurrentUnit < handle
    %% RecurrentUnit: vanilla RNN cell with leaky-ReLU hidden and output layers
    % Supports truncated BPTT, moment storage for Adam, and optional caching for training
    properties
        inSize
        outSize
        biases
        weights
        hState
        dLdh
        actCache
        preactCache
        dW, db
        vdw, sdw
        vdb, sdb
        a

    end
    
    %%
    methods
        %% RecurrentUnit Constructor: initialize parameters and optimizer state
        function obj = RecurrentUnit(inputSize, hSize, outputSize)
            
            obj.inSize      = inputSize;
            obj.outSize     = outputSize;

            % Initialize weight matrices with He-normal scaling
            obj.weights     = cell(1,2);
            obj.weights{1}  = randn(hSize,inputSize+hSize) * sqrt(2/(inputSize+hSize));  % Wx & Wh
            obj.weights{2}  = randn(outputSize,hSize)      * sqrt(2/hSize);              % Wy

            % Initialize biases to offset mean of incoming weights
            obj.biases      = cell(1,2);
            obj.biases{1}   = -mean(obj.weights{1},2);    % bh
            obj.biases{2}   = -mean(obj.weights{2},2);    % by

            % Initialize hidden state
            obj.hState      = zeros(hSize, 1);

            % accumulated backprop gradient dE/dh for previous timestep
            obj.dLdh        = zeros(size(obj.hState,1), size(obj.hState,2));

            % Leaky-ReLU parameter
            obj.a           = 0.01;

            % Initialize caches and gradients
            obj.actCache    = [];
            obj.preactCache = [];
            obj.dW          = cell(size(obj.weights));
            obj.db          = cell(size(obj.biases));

            for j = 1:numel(obj.weights)
                obj.dW{j}  = zeros(size(obj.weights{j}));
            end
            for j = 1:numel(obj.biases)
                obj.db{j}  = zeros(size(obj.biases{j}));
            end

            % Initialize Adam moment buffers
            for j = 1:numel(obj.weights)
                obj.vdw{j} = zeros(size(obj.weights{j}));
                obj.sdw{j} = zeros(size(obj.weights{j}));
                obj.dW{j}  = zeros(size(obj.weights{j}));
                obj.vdb{j} = zeros(size(obj.biases{j}));
                obj.sdb{j} = zeros(size(obj.biases{j}));
                obj.db{j}  = zeros(size(obj.biases{j}));
            end
        end

        %% Forward: single time-step. Optionally cache history if 'train' flag
        function a_out = forward(obj, x_in, numSteps, t, varargin)
            % Inputs:
            %   x_in     - backpropagated error from the next layer, [1 × inSize] 
            %   numSteps - length of the training sequence
            %   t        - time index of the cached activations
            % Outputs:
            %   a_out    - backpropagated error to the previous layer, [1 × outSize]

            doCache = ~isempty(varargin) && strcmp(varargin{1}, 'train');

            % columnize input for matrix ops
            x_in    = x_in';
            
            % compute hidden pre-activation and activation
            zh      = obj.weights{1} * [x_in; obj.hState(:,t)] + obj.biases{1};
            h       = ReLU(zh, obj.a);

            % compute output pre-activation and activation
            z       = obj.weights{2} * h + obj.biases{2};
            a_out   = ReLU(z, obj.a);

            % convert back to row form
            a_out   = a_out';

            if doCache
                if isempty(obj.actCache)
                    % initialize activation and hidden state cache
                    obj.hState       = zeros(length(h),numSteps+1);
                    obj.actCache     = zeros(length(x_in),numSteps);
                    obj.preactCache  = zeros(length(z),numSteps);
                end
                % store activations and hidden states for truncated BPTT
                obj.hState(:,t+1)    = h;
                obj.actCache(:,t)    = x_in';
                obj.preactCache(:,t) = z;
            else
                % or keep only the last state for inference
                obj.hState           = h;
            end
        end

        %% Backprop: one-step truncated BPTT update
        function [d_in, dW_new, db_new] = backprop(obj, d_out, t)
            % Inputs:
            %   d_out  - backpropagated error from the next layer, [1 × outSize]
            %   t      - time index of the cached activations
            % Outputs:
            %   d_in   - backpropagated error at the input, [1 × inSize]
            %   dW_new - update vector of weights from timestep t
            %   db_new - update vector of biases from timestep t

            % Preallocate gradients
            db_new    = cell(1,2);
            dW_new    = cell(1,2);

            % Values from previous timesteps
            h         = obj.hState(:,t+1);
            h_prev    = obj.hState(:,  t);
            a_in      = obj.actCache(:,t);
        
            delta     = d_out' .* ReLU_prime(obj.preactCache(:,t), obj.a);
        
            % error coming from the output
            dLdh_from_y = obj.weights{2}' * delta;
        
            % total error w.r.t hidden activation at t
            % error from the output + error from the next timestep
            dL_dh     = dLdh_from_y + obj.dLdh;
        
            % backprop through ReLU at hidden layer
            z0        = obj.weights{1} * [a_in; h_prev] + obj.biases{1};
            delta_h   = dL_dh .* ReLU_prime(z0, obj.a);  % ∂E/∂z0
        
            % Gradients w.r.t. hidden‐layer and output-layer parameters
            dW_new{1} = delta_h * [a_in; h_prev]';  % ∂E/∂Wx & ∂E/∂Wh
            dW_new{2} = delta * h';                 % ∂E/∂Wy
            db_new{1} = delta_h;                    % ∂E/∂bh
            db_new{2} = delta;                      % ∂E/∂by
        
            % Package errors for next step
            obj.dLdh  = obj.weights{1}(:,obj.inSize+1:end)' * delta_h;    % push ∂E/∂h_prev back in time
            d_in      = obj.weights{1}(:,1:obj.inSize)' * delta_h;        % propagate error into input
            d_in      = d_in';                                               % back to row form
        end

        %% ApplyAdam: update weights and biases with Adam
        function applyAdam(obj, eta, beta1, beta2, m, t, eps)
            [obj.weights, obj.vdw, obj.sdw] = adamUpdate(obj.weights, obj.dW, obj.vdw, obj.sdw, beta1, beta2, t, eta, m, eps);
            [obj.biases,  obj.vdb, obj.sdb] = adamUpdate(obj.biases,  obj.db, obj.vdb, obj.sdb, beta1, beta2, t, eta, m, eps);
        end

        %% Resets
        function resetMemory(obj)
            % Clear hidden-state history, BPTT gradient accumulator and stored activations
            obj.hState      = zeros(length(obj.biases{1}),1);
            obj.dLdh        = zeros(size(obj.hState));
            obj.actCache    = [];
            obj.preactCache = [];
        end

        function resetGrads(obj)
            % Zero accumulated gradients before new batch
            for j = 1:numel(obj.weights)
                obj.dW{j} = zeros(size(obj.weights{j}));
            end
            for j = 1:numel(obj.biases)
                obj.db{j} = zeros(size(obj.biases{j}));
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