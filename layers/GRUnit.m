classdef GRUnit < handle
    %% GRUnit: Gated recurrent unit with sigmoid and tanh activations
    % Supports truncated BPTT, moment storage for Adam, and optional caching for training
    properties
        inSize
        outSize
        biases
        weights
        hiddenState
        dLdh
        acts
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
            
            obj.inSize  = inSize;
            obj.outSize = outSize;

            % Initialize weight matrices with He-normal scaling
            obj.weights    = cell(1,3);
            obj.weights{1} = randn(outSize,inSize+outSize)*sqrt(2/(outSize*(inSize+outSize)));  % Wxz & Whz
            obj.weights{2} = randn(outSize,inSize+outSize)*sqrt(2/(outSize*(inSize+outSize)));  % Wxr & Whr 
            obj.weights{3} = randn(outSize,inSize+outSize)*sqrt(2/(outSize*(inSize+outSize)));         % Wxh & Whh

            % Initialize biases to offset mean of incoming weights
            obj.biases    = cell(1,3);
            obj.biases{1} = -mean(obj.weights{1},2);    % bz
            obj.biases{2} = -mean(obj.weights{2},2);    % br
            obj.biases{3} = -mean(obj.weights{3},2);    % bh

            % Initialize hidden state
            obj.hiddenState = zeros(outSize, 1);
            
            % accumulated backprop gradient dE/dh for previous time step
            obj.dLdh        = zeros(size(obj.hiddenState,1), size(obj.hiddenState,2));

            % Leaky-ReLU parameter
            obj.a = 0.01;

            % Initialize caches and gradients
            obj.acts  = [];
            obj.zs    = [];
            obj.rs    = [];
            obj.hhats = [];
            obj.dW    = cell(size(obj.weights));
            obj.db    = cell(size(obj.biases));

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
        function a_out = forward(obj, x_in, varargin)
            % x_in  - input data, row vector [1×inputSize]
            % a_out - output data, row vector [1×outputSize]

            doCache = ~isempty(varargin) && strcmp(varargin{1}, 'train');

            % columnize input for matrix ops
            x_in   = x_in';
            
            z_pre = obj.weights{1} * [x_in; obj.hiddenState(:,end)] + obj.biases{1};
            z = sigm(z_pre);

            r_pre = obj.weights{2} * [x_in; obj.hiddenState(:,end)] + obj.biases{2};
            r = sigm(r_pre);

            hhat = tanh(obj.weights{3} * [x_in; r.*obj.hiddenState(:,end)]);

            h = z .* obj.hiddenState(:,end) + (1 - z) .* hhat;

            % convert back to row form
            a_out = h';

            if doCache
                % append to history for BPTT...
                obj.hiddenState = [obj.hiddenState, h];
                obj.acts        = [obj.acts, x_in];
                obj.zs          = [obj.zs, z];
                obj.rs          = [obj.rs, r];
                obj.hhats       = [obj.hhats, hhat];
            else
                % or keep only the last state for inference
                obj.hiddenState = h;
            end
        end

        %% Backprop: one-step truncated BPTT update
        function d_in = backprop(obj, d_out, t_idx)
            % d_out - gradient from next layer, row [1×outputSize] 
            % t_idx - time index to reference the stored history
            % d_in  - gradient to previous layer, row [1×inputSize] 

            % Preallocate gradients
            db_new = cell(1,3);
            dW_new = cell(1,3);

            d_out = d_out';

            h_prev = obj.hiddenState(:,t_idx);
            xt     = obj.acts(:,t_idx);
            zt     = obj.zs(:,t_idx);
            rt     = obj.rs(:,t_idx);
            hhatt  = obj.hhats(:,t_idx);

            Wz = obj.weights{1}(:,1:obj.inSize);
            Wr = obj.weights{2}(:,1:obj.inSize);
            Wh = obj.weights{3}(:,1:obj.inSize);
            Uz = obj.weights{1}(:,obj.inSize+1:end);
            Ur = obj.weights{2}(:,obj.inSize+1:end);
            Uh = obj.weights{3}(:,obj.inSize+1:end);

            delta   = d_out + obj.dLdh;
            delta_z = delta .* (h_prev - hhatt) .* zt .* (1 - zt);
            delta_h = delta .* (1-zt) .* (1-hhatt.^2);
            % delta_r = Uh * delta_h .* h_prev .* rt.* (1-rt);
            delta_r = Uh' * delta_h .* h_prev .* rt .* (1-rt);
            
        
            % Gradients w.r.t. hidden‐layer and output-layer parameters
            dW_new{1} = delta_z * [xt; h_prev]';
            dW_new{2} = delta_r * [xt; h_prev]';
            dW_new{3} = delta_h * [xt; rt .* h_prev]';      % ∂E/∂Wx
            
            db_new{1} = delta_z;
            db_new{2} = delta_r;
            db_new{3} = delta_h;             % ∂E/∂bh
            
        
            % Package errors for next step
            obj.dLdh = delta.*zt + Uz'*delta_z + Uh'*(delta_h.*rt) + Ur'*delta_r;  % push ∂E/∂h_prev back in time
            d_in  = Wz'*delta_z + Wh'*delta_h + Wr'*delta_r;                       % propagate error into input
            d_in  = d_in';                                                         % back to row form

            % accumulate gradients
            obj.dW = cellfun(@(x,y) x + y, obj.dW, dW_new, 'UniformOutput', false);
            obj.db = cellfun(@(x,y) x + y, obj.db, db_new, 'UniformOutput', false);
       
        end

        %% ApplyAdam: update weights and biases with Adam
        function applyAdam(obj, eta, beta1, beta2, m, t, eps)
            [obj.weights, obj.vdw, obj.sdw] = adamUpdate(obj.weights, obj.dW, obj.vdw, obj.sdw, beta1, beta2, t, eta, m, eps);
            [obj.biases,  obj.vdb, obj.sdb] = adamUpdate(obj.biases,  obj.db, obj.vdb, obj.sdb, beta1, beta2, t, eta, m, eps);
        end

        %% Resets
        function resetMemory(obj)
            % Clear hidden-state history, BPTT gradient accumulator and stored activations
            obj.hiddenState = zeros(length(obj.biases{1}),1);
            obj.dLdh        = zeros(size(obj.hiddenState,1), size(obj.hiddenState,2));
            obj.acts  = [];
            obj.zs    = [];
            obj.rs    = [];
            obj.hhats = [];
        end

        function resetGrads(obj)
            % Zero accumulated gradients before new batch
            for j = 1:numel(obj.weights)
                obj.dW{j}  = zeros(size(obj.weights{j}));
            end
            for j = 1:numel(obj.biases)
                obj.db{j}  = zeros(size(obj.biases{j}));
            end
        end
    end
end


%%
function s = sigm(z)
    s = 1./(1+exp(-z));
end