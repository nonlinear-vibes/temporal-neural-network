classdef PoolingLayer < handle
    %% PoolingLayer: 1D temporal max-pooling over consecutive frames
    % Reduces time dimension by poolingRatio, retains channel and kernel dims
    properties
        poolingRatio
        weights
        biases
        acts
    end
    
    %%
    methods
        %% PoolingLayer constructor
        function obj = PoolingLayer(poolingRatio)
            % Set pooling factor, initialize dummy params
            obj.poolingRatio = poolingRatio; % downsampling factor along time axis
            obj.weights      = 1;            % dummy for Adam compatibility (unused)
            obj.biases       = 0;            % dummy for Adam compatibility (unused)
            obj.acts         = [];           % stores last input for backprop
        end
        
        %% Forward: max over non-overlapping windows along time
        function as = forward(obj, input, varargin)
            % input: [T×C×K] tensor (time×channels×kernels)

            doCache = ~isempty(varargin) && strcmp(varargin{1}, 'train');

            % Compute output size after pooling
            outputLength = floor((size(input,1)/obj.poolingRatio));
            sizes        = size(input);
            sizes(1)     = outputLength;
            as           = zeros(sizes);
            zs           = zeros(sizes);

            % Slide window and take max across time slice
            for i = 1:outputLength
                startIdx   = (i-1)*obj.poolingRatio+1;
                endIdx     = i*obj.poolingRatio;
                inputSlice = input(startIdx:endIdx, :, :);
                as(i,:,:)  = squeeze(max(inputSlice, [], 1));
                zs(i,:,:)  = squeeze(max(inputSlice, [], 1));
            end

            % Store activations and preactivations for backprop
            if doCache
                obj.acts = input;
            end
        end

        %% Backprop: route gradients only to max positions
        function d_input = backprop(obj, d_output)

            % Prepare input gradient buffer
            d_input            = zeros(size(obj.acts));
            d_output_augmented = zeros(size(d_input));
            argmaxMask         = zeros(size(d_input));

            % For each pooled frame, find argmax locations
            for i = 1:size(d_output,1)
                [~, argmax_idx] = max(obj.acts(((i-1)*obj.poolingRatio+1:i*obj.poolingRatio),:,:), [], 1);
                d_output_augmented(((i-1)*obj.poolingRatio+1:i*obj.poolingRatio),:,:) = repmat(d_output(i,:,:),[obj.poolingRatio,1,1]);

                % Mark positions where idx == p
                for j = 1:obj.poolingRatio
                    argmaxMask((i-1)*obj.poolingRatio+j,argmax_idx == j) = 1;
                end
            end
            argmaxMask = logical(argmaxMask);

            % Assign gradients only at max locations
            d_input(argmaxMask) = d_output_augmented(argmaxMask);
        end

        %% ApplyAdam: only for consistency
        function applyAdam(obj, varargin)
            obj.weights = obj.weights;
            obj.biases  = obj.biases;
        end

        %% ResetStoredActivations: clear cached input
        function resetStoredActivations(obj)
            obj.acts         = [];
        end

    end
end