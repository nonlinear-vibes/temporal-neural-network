classdef LSTMUnit < handle
    %% LSTMUnit: Long short-term memory unit with sigmoid and tanh activations
    % Supports truncated BPTT, moment storage for Adam, and optional caching for training
    properties
        inSize
        outSize
        biases
        weights
        hShort
        cLong
        dLdh
        dLdc
        actsCache
        inputCache
        dW, db
        vdw, sdw
        vdb, sdb

    end
    
    %%
    methods
        %% GRUnit Constructor: initialize parameters and optimizer state
        function obj = LSTMUnit(inSize, outSize)
            
            obj.inSize  = inSize;
            obj.outSize = outSize;

            % Initialize weight matrices with He-normal scaling
            obj.weights = randn(4*outSize,inSize+outSize)*sqrt(2/(outSize*(inSize+outSize)));
            obj.biases  = -mean(obj.weights,2);

            % Initialize hidden state
            obj.hShort = zeros(outSize, 1);
            obj.cLong  = zeros(outSize, 1);
            
            % accumulated backprop gradient dE/dh for previous time step
            obj.dLdh = zeros(4*outSize, 1);
            obj.dLdc = zeros(outSize, 1);

            % Initialize caches and gradients
            obj.actsCache  = [];
            obj.inputCache = [];

            obj.dW = cell(size(obj.weights));
            obj.db = cell(size(obj.biases));

            % Initialize Adam moment buffers
            obj.vdw = zeros(size(obj.weights));
            obj.sdw = zeros(size(obj.weights));
            obj.dW  = zeros(size(obj.weights));
            obj.vdb = zeros(size(obj.biases));
            obj.sdb = zeros(size(obj.biases));
            obj.db  = zeros(size(obj.biases));
        end

        %% Forward: single time-step. Optionally cache history if 'train' flag
        function a_out = forward(obj, x_in, varargin)
            % x_in  - input data, row vector [1×inputSize]
            % a_out - output data, row vector [1×outputSize]

            doCache = ~isempty(varargin) && strcmp(varargin{1}, 'train');

            % columnize input for matrix ops
            x_in   = x_in';
            c_prev = obj.cLong(:,end);
            h_prev = obj.hShort(:,end);
            
            preacts = obj.weights * [x_in; h_prev] + obj.biases;
            acts    = sigm(preacts(1:3*obj.outSize));
            c_tilde = tanh(preacts(3*obj.outSize+1:end));

            c = acts(1:obj.outSize) .* c_prev + acts(obj.outSize+1:2*obj.outSize) .* c_tilde;
            h = acts(2*obj.outSize+1:end) .* tanh(c);

            % convert back to row form
            a_out = h';

            if doCache
                % append to history for BPTT...
                obj.hShort     = [obj.hShort, h];
                obj.cLong      = [obj.cLong,  c];
                obj.actsCache  = [obj.actsCache, [acts; c_tilde]];
                obj.inputCache = [obj.inputCache, x_in];
            else
                % or keep only the last state for inference
                obj.hShort = h;
                obj.cLong  = c;
            end
        end

        %% Backprop: one-step BPTT update
        function d_in = backprop(obj, d_out, t_idx)

            d_out  = d_out';

            h_prev = obj.hShort(:,t_idx);
            c_prev = obj.cLong(:,t_idx);
            c_t    = obj.cLong(:,t_idx+1);
            x_t    = obj.inputCache(:,t_idx);

            f_t = obj.actsCache(1:obj.outSize,t_idx);
            i_t = obj.actsCache(obj.outSize+1:2*obj.outSize,t_idx);
            o_t = obj.actsCache(2*obj.outSize+1:3*obj.outSize,t_idx);
            c_tilde_t =obj.actsCache(3*obj.outSize+1:end,t_idx);

            dLdh_tot = d_out + obj.weights(:,obj.inSize+1:end)' * obj.dLdh;
            delta    = dLdh_tot .* o_t .* (1 - tanh(c_t).^2) + obj.dLdc;

            delta_o = dLdh_tot .* tanh(c_t) .* o_t .* (1 - o_t);
            delta_f = delta .* c_prev .* f_t .* (1 - f_t);
            delta_i = delta .* c_tilde_t .* i_t .* (1 - i_t);
            delta_c = delta .* f_t;

            dLdh_tot_c = d_out + obj.weights(1:3*obj.outSize,obj.inSize+1:end)' * obj.dLdh(1:3*obj.outSize);
            delta_ctilde = (dLdh_tot_c .* o_t .* (1 - tanh(c_t).^2) + obj.dLdc) .* i_t .* (1 - c_tilde_t.^2);
            % delta_ctilde = delta .* i_t .* (1 - i_t);
        
            dW_update = [delta_f; delta_i; delta_o; delta_ctilde] * [x_t; h_prev]';
            db_update = [delta_f; delta_i; delta_o; delta_ctilde];        
        
            % Package errors for next step
            obj.dLdh = [delta_f; delta_i; delta_o; delta_ctilde];  
            obj.dLdc = delta_c;

            d_in  = obj.weights(:,1:obj.inSize)' * obj.dLdh;                       
            d_in  = d_in';                                                  

            % accumulate gradients
            obj.dW = obj.dW + dW_update;
            obj.db = obj.db + db_update;
       
        end

        %% ApplyAdam: update weights and biases with Adam
        function applyAdam(obj, eta, beta1, beta2, m, t, eps)
            [obj.weights, obj.vdw, obj.sdw] = adamUpdate(obj.weights, obj.dW, obj.vdw, obj.sdw, beta1, beta2, t, eta, m, eps);
            [obj.biases,  obj.vdb, obj.sdb] = adamUpdate(obj.biases,  obj.db, obj.vdb, obj.sdb, beta1, beta2, t, eta, m, eps);
        end

        %% Resets
        function resetMemory(obj)
            % Clear hidden-state history, BPTT gradient accumulator and stored activations
            obj.hShort = zeros(obj.outSize,1);
            obj.cLong  = zeros(obj.outSize,1);
            obj.dLdh   = zeros(4*obj.outSize,1);
            obj.dLdc   = zeros(obj.outSize,1);

            obj.actsCache  = [];
            obj.inputCache = [];
        end

        function resetGrads(obj)
            % Zero accumulated gradients before new batch
            for j = 1:numel(obj.weights)
                obj.dW = zeros(size(obj.weights));
            end
            for j = 1:numel(obj.biases)
                obj.db = zeros(size(obj.biases));
            end
        end
    end
end


%%
function s = sigm(z)
    s = 1./(1+exp(-z));
end