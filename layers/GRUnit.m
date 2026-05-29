classdef GRUnit < handle
    %% GRUnit: Gated recurrent unit with sigmoid and tanh activations
    % Supports truncated BPTT, moment storage for Adam, and optional caching for training
    properties
        inSize
        outSize
        biases
        weights
        hState
        dLdh
        preactCache
        zs, rs, hhats
        dW, db
        vdw, sdw
        vdb, sdb
        a

    end
    
    %%
    methods
        %% GRUnit Constructor: initialize parameters and optimizer state
        function obj = GRUnit(inSize, outSize)
            
            obj.inSize      = inSize;
            obj.outSize     = outSize;

            % Initialize weight matrices with He-normal scaling
            obj.weights{1}  = randn(outSize,inSize+outSize)*sqrt(2/(inSize+outSize));  % [Wxz, Whz]
            obj.weights{2}  = randn(outSize,inSize+outSize)*sqrt(2/(inSize+outSize));  % [Wxr, Whr]
            obj.weights{3}  = randn(outSize,inSize+outSize)*sqrt(2/(inSize+outSize));  % [Wxh, Whh]

            % Initialize biases to offset mean of incoming weights
            obj.biases{1}   = -mean(obj.weights{1},2);    % bz
            obj.biases{2}   = -mean(obj.weights{2},2);    % br
            obj.biases{3}   = -mean(obj.weights{3},2);    % bh

            % Initialize hidden state
            obj.hState      = zeros(outSize, 1);
            
            % accumulated backprop gradient dE/dh for previous time step
            obj.dLdh        = zeros(size(obj.hState,1), size(obj.hState,2));

            % Leaky-ReLU parameter
            obj.a           = 0.01;

            % Initialize caches and gradients
            obj.preactCache = [];
            obj.zs          = [];
            obj.rs          = [];
            obj.hhats       = [];

            % Initialize Adam moment buffers
            for j = 1:numel(obj.weights)
                obj.vdw{j}  = zeros(size(obj.weights{j}));
                obj.sdw{j}  = zeros(size(obj.weights{j}));
                obj.dW{j}   = zeros(size(obj.weights{j}));
                obj.vdb{j}  = zeros(size(obj.biases{j}));
                obj.sdb{j}  = zeros(size(obj.biases{j}));
                obj.db{j}   = zeros(size(obj.biases{j}));
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
            x_in  = x_in';
            
            z_pre = obj.weights{1} * [x_in; obj.hState(:,t)] + obj.biases{1};
            z     = sigm(z_pre);

            r_pre = obj.weights{2} * [x_in; obj.hState(:,t)] + obj.biases{2};
            r     = sigm(r_pre);

            hhat  = tanh(obj.weights{3} * [x_in; r.*obj.hState(:,t)] + obj.biases{3});

            h     = z .* obj.hState(:,t) + (1 - z) .* hhat;

            % convert back to row form
            a_out = h';

            if doCache
                if isempty(obj.preactCache)
                    % initialize activation and hidden state cache
                    obj.hState       = zeros(length(h),numSteps+1);
                    obj.preactCache  = zeros(length(x_in),numSteps);
                    obj.zs           = zeros(length(z),numSteps);
                    obj.rs           = zeros(length(r),numSteps);
                    obj.hhats        = zeros(length(hhat),numSteps+1);
                end
                % append to history for BPTT...
                obj.hState(:,t+1)    = h;
                obj.preactCache(:,t) = x_in;
                obj.zs(:,t)          = z;
                obj.rs(:,t)          = r;
                obj.hhats(:,t+1)     = hhat;
            else
                % or keep only the last state for inference
                obj.hState = h;
            end
        end

        %% Backprop: one-step truncated BPTT update
        function [d_in, dW_new, db_new] = backprop(obj, d_out, t)
            % Inputs:
            %   d_out  - backpropagated error from the next layer, [1 × outSize]
            %   t      - time index of the cached activations
            % Outputs:
            %   d_in   - backpropagated error to the previous layer, [1 × inSize]
            %   dW_new - update vector of weights from timestep t
            %   db_new - update vector of biases from timestep t

            % Preallocate gradients
            db_new  = cell(1,3);
            dW_new  = cell(1,3);

            d_out   = d_out';

            h_prev  = obj.hState(:,t);
            xt      = obj.preactCache(:,t);
            zt      = obj.zs(:,t);
            rt      = obj.rs(:,t);
            hhatt   = obj.hhats(:,t);

            Wz      = obj.weights{1}(:,1:obj.inSize);
            Wr      = obj.weights{2}(:,1:obj.inSize);
            Wh      = obj.weights{3}(:,1:obj.inSize);
            Uz      = obj.weights{1}(:,obj.inSize+1:end);
            Ur      = obj.weights{2}(:,obj.inSize+1:end);
            Uh      = obj.weights{3}(:,obj.inSize+1:end);

            delta   = d_out + obj.dLdh;
            delta_z = delta .* (h_prev - hhatt) .* zt .* (1 - zt);
            delta_h = delta .* (1-zt) .* (1-hhatt.^2);
            delta_r = Uh' * delta_h .* h_prev .* rt .* (1-rt);
            
        
            % Gradients w.r.t. hidden‐layer and output-layer parameters
            % ∂E/∂Wx
            dW_new{1} = delta_z * [xt; h_prev]';
            dW_new{2} = delta_r * [xt; h_prev]';
            dW_new{3} = delta_h * [xt; rt .* h_prev]';
            
            % ∂E/∂b
            db_new{1} = delta_z;
            db_new{2} = delta_r;
            db_new{3} = delta_h;             
            
            % Package errors for next step
            obj.dLdh  = delta.*zt + Uz'*delta_z + Uh'*(delta_h.*rt) + Ur'*delta_r;  % push ∂E/∂h_prev back in time
            d_in      = Wz'*delta_z + Wh'*delta_h + Wr'*delta_r;                    % propagate error into input
            d_in      = d_in';                                                      % back to row form
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
            obj.preactCache = [];
            obj.zs          = [];
            obj.rs          = [];
            obj.hhats       = [];
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
function s = sigm(z)
    z = max(min(z,50),-50); % numerical stabilization
    s = 1./(1+exp(-z));
end