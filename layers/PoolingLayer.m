classdef PoolingLayer < handle
    %% PoolingLayer: 1D temporal max-pooling over consecutive frames
    % Reduces time dimension by poolingRatio, retains channel and kernel dims
    properties
        poolingRatio
        actCache
    end
    
    %%
    methods
        %% PoolingLayer constructor
        function obj = PoolingLayer(poolingRatio)
            % Set pooling factor, initialize dummy params
            obj.poolingRatio = poolingRatio; % downsampling factor along time axis
            obj.actCache     = [];           % stores last input for backprop
        end
        
        %% Forward: max over non-overlapping windows along time
        function a_out = forward(obj, x_in, isTraining)
            % Inputs:
            %   x_in     - input data tensor [numSteps × numChannels × inFeatures]
            %   varargin - if 'train' is given, activations and preactivations are stored 
            % Output:
            %   a_out    - output activation tensor [{poolingRatio * numSteps} × numChannels × inFeatures] 

            % Compute output size after pooling
            outputLength = floor(size(x_in,1)/obj.poolingRatio);
            sizes        = size(x_in);
            sizes(1)     = outputLength;
            a_out        = zeros(sizes);

            % Slide window and take max across time slice
            for i = 1:outputLength
                startIdx     = (i-1)*obj.poolingRatio+1;
                endIdx       = i*obj.poolingRatio;
                inputSlice   = x_in(startIdx:endIdx, :, :);
                a_out(i,:,:) = max(inputSlice, [], 1);
            end

            % Store activations and preactivations for backprop
            if isTraining
                obj.actCache = x_in;
            end
        end

        %% Backprop: route gradients only to max positions
        function [d_in, dW_new, db_new] = backprop(obj, d_out)
            % Inputs:
            %   d_out  - backpropagated error at the output, [outputSteps × numChannels × numFeatures]
            % Outputs:
            %   d_in   - backpropagated error at the input, [{poolingRatio * outputSteps} × numChannels × numFeatures]
            %   dW_new - empty variable for uniform interface
            %   db_new - empty variable for uniform interface

            % Prepare input gradient buffer
            d_in               = zeros(size(obj.actCache));
            d_output_augmented = zeros(size(d_in));
            argmaxMask         = zeros(size(d_in));

            % For each pooled frame, find argmax locations
            for i = 1:size(d_out,1)
                [~, argmax_idx] = max(obj.actCache(((i-1)*obj.poolingRatio+1:i*obj.poolingRatio),:,:), [], 1);
                d_output_augmented(((i-1)*obj.poolingRatio+1:i*obj.poolingRatio),:,:) = repmat(d_out(i,:,:),[obj.poolingRatio,1,1]);

                % Mark positions where idx == p
                for j = 1:obj.poolingRatio
                    argmaxMask((i-1)*obj.poolingRatio+j,argmax_idx == j) = 1;
                end
            end
            argmaxMask = logical(argmaxMask);

            % Assign gradients only at max locations
            d_in(argmaxMask) = d_output_augmented(argmaxMask);
            
            % Dummies for compatibility
            dW_new = [];
            db_new = [];
        end

        %% ResetStoredActivations: clear cached input
        function resetStoredActivations(obj)
            obj.actCache = [];
        end
    end
end