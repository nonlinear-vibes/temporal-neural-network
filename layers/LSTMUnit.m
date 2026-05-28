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
        acts
        input
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
            obj.weights = {randn(4*outSize,inSize+outSize)*sqrt(2/(inSize+outSize))};
            obj.biases  = {-mean(obj.weights{1},2)};

            % Initialize hidden state
            obj.hShort = zeros(outSize, 1);
            obj.cLong  = zeros(outSize, 1);
            
            % accumulated backprop gradient dE/dh for previous time step
            obj.dLdh = zeros(4*outSize, 1);
            obj.dLdc = zeros(outSize, 1);

            % Initialize caches and gradients
            obj.acts  = [];
            obj.input = [];

            % Initialize Adam moment buffers
            obj.vdw = {zeros(size(obj.weights{1}))};
            obj.sdw = {zeros(size(obj.weights{1}))};
            obj.dW  = {zeros(size(obj.weights{1}))};
            obj.vdb = {zeros(size(obj.biases{1}))};
            obj.sdb = {zeros(size(obj.biases{1}))};
            obj.db  = {zeros(size(obj.biases{1}))};
        end

        %% Forward: single time-step. Optionally cache history if 'train' flag
        function a_out = forward(obj, x_in, numSteps, t, varargin)
            % x_in  - input data, row vector [1×inputSize]
            % a_out - output data, row vector [1×outputSize]

            doCache = ~isempty(varargin) && strcmp(varargin{1}, 'train');

            % columnize input for matrix ops
            x_in   = x_in';
            c_prev = obj.cLong(:,t);
            h_prev = obj.hShort(:,t);
            
            preacts = obj.weights{1} * [x_in; h_prev] + obj.biases{1};
            as      = sigm(preacts(1:3*obj.outSize));
            c_tilde = tanh(preacts(3*obj.outSize+1:end));

            c = as(1:obj.outSize) .* c_prev + as(obj.outSize+1:2*obj.outSize) .* c_tilde;
            h = as(2*obj.outSize+1:end) .* tanh(c);

            % convert back to row form
            a_out = h';

            if doCache
                if isempty(obj.acts)
                    % initialize activation and hidden state cache
                    obj.hShort = zeros(length(h),numSteps+1);
                    obj.cLong  = zeros(length(c),numSteps+1);
                    obj.acts   = zeros(length(as)+length(c_tilde),numSteps);
                    obj.input  = zeros(length(x_in),numSteps);
                end
                % append to history for BPTT...
                obj.hShort(:,t+1) = h;
                obj.cLong(:,t+1)  = c;
                obj.acts(:,t)     = [as; c_tilde];
                obj.input(:,t)    = x_in;
            else
                % or keep only the last state for inference
                obj.hShort = h;
                obj.cLong  = c;
            end
        end

        %% Backprop: one-step BPTT update
        function [d_in, dW_update, db_update] = backprop(obj, d_out, t_idx)
            % d_out - gradient from next layer, row [1×outputSize] 
            % t_idx - time index to reference the stored history
            % d_in  - gradient to previous layer, row [1×inputSize] 

            d_out  = d_out';

            h_prev = obj.hShort(:,t_idx);
            c_prev = obj.cLong(:,t_idx);
            c_t    = obj.cLong(:,t_idx+1);
            x_t    = obj.input(:,t_idx);

            f_t = obj.acts(1:obj.outSize,t_idx);
            i_t = obj.acts(obj.outSize+1:2*obj.outSize,t_idx);
            o_t = obj.acts(2*obj.outSize+1:3*obj.outSize,t_idx);
            c_tilde_t =obj.acts(3*obj.outSize+1:end,t_idx);

            dLdh_tot = d_out + obj.weights{1}(:,obj.inSize+1:end)' * obj.dLdh;
            delta    = dLdh_tot .* o_t .* (1 - tanh(c_t).^2) + obj.dLdc;

            delta_o = dLdh_tot .* tanh(c_t) .* o_t .* (1 - o_t);
            delta_f = delta .* c_prev .* f_t .* (1 - f_t);
            delta_i = delta .* c_tilde_t .* i_t .* (1 - i_t);
            delta_c = delta .* f_t;

            dLdh_tot_c = d_out + obj.weights{1}(1:3*obj.outSize,obj.inSize+1:end)' * obj.dLdh(1:3*obj.outSize);
            delta_ctilde = (dLdh_tot_c .* o_t .* (1 - tanh(c_t).^2) + obj.dLdc) .* i_t .* (1 - c_tilde_t.^2);
        
            dW_update = {[delta_f; delta_i; delta_o; delta_ctilde] * [x_t; h_prev]'};
            db_update = {[delta_f; delta_i; delta_o; delta_ctilde]};        
        
            % Package errors for next step
            obj.dLdh = [delta_f; delta_i; delta_o; delta_ctilde];  
            obj.dLdc = delta_c;

            d_in  = obj.weights{1}(:,1:obj.inSize)' * obj.dLdh;                       
            d_in  = d_in';                                                  
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

            obj.acts  = [];
            obj.input = [];
        end

        function resetGrads(obj)
        % Zero accumulated gradients before new batch
            obj.dW = {zeros(size(obj.weights{1}))};
            obj.db = {zeros(size(obj.biases{1}))};
        end
    end
end


%%
function s = sigm(z)
    z = max(min(z,50),-50); % numerical stabilization
    s = 1./(1+exp(-z));
end