classdef ConvolutionalLayer < handle
    %% ConvolutionalLayer: separate kernels scanning through each channel
    properties
        numChannels
        inFeatures
        outFeatures
        kernelSize
        weights
        biases
        actCache
        preactCache
        dW, db
        vdw, vdb
        sdw, sdb
        a
        
    end
    
    %%
    methods
        %% ConvolutionalLayer constructor
        function obj = ConvolutionalLayer(numChannels, inFeatures, outFeatures, kernelSize)
            obj.numChannels = numChannels;  % number of input channels
            obj.inFeatures  = inFeatures;   % number of feature maps input (depth)
            obj.outFeatures = outFeatures;  % number of output kernels per channel
            obj.kernelSize  = kernelSize;   % length of the each kernel
            
            % He initialization for conv filters
            scale       = sqrt(2/(kernelSize*inFeatures));
            obj.weights = randn(kernelSize, numChannels, outFeatures, inFeatures) * scale;

            % Initialize biases to negative mean of weights
            meanWeights = reshape(mean(obj.weights,1), [numChannels, outFeatures, inFeatures]);
            biasValues  = mean(meanWeights,3);              % [numChannels x outFeatures]
            obj.biases  = -biasValues;

            % Initialize Adam moments
            obj.vdw = zeros(size(obj.weights));
            obj.sdw = zeros(size(obj.weights));
            obj.vdb = zeros(size(obj.biases));
            obj.sdb = zeros(size(obj.biases));

            % Initialize gradient buffers
            obj.dW = zeros(size(obj.weights));
            obj.db = zeros(size(obj.biases));

            % Leaky-ReLU parameter
            obj.a = 0.01;
        end
        
        %% Forward: 1D convolutions over time
        function a_out = forward(obj, x_in, isTraining)
            % Inputs:
            %   x_in       - input data tensor [T × numChannels × inFeatures]
            %   isTraining - if true, activations and preactivations are stored 
            % Output:
            %   a_out      - output activation tensor [numSteps × numChannels × outFeatures] 

            numSteps = size(x_in,1) - obj.kernelSize + 1;
            a_out    = zeros(numSteps, obj.numChannels, obj.outFeatures);
            zs       = zeros(numSteps, obj.numChannels, obj.outFeatures);
        
            for i = 1:numSteps
                xSlice = x_in(i:i+obj.kernelSize-1,:,:);   % [kernelSize x numChannels x inFeatures]
                for k = 1:obj.outFeatures
                    % Reshape to the known target shape instead of squeeze(),
                    % which would silently drop numChannels or inFeatures too
                    % if either equals 1.
                    Wk = reshape(obj.weights(:,:,k,:), [obj.kernelSize, obj.numChannels, obj.inFeatures]);
                    dotproduct = xSlice .* Wk;                                   % [kernelSize x numChannels x inFeatures]
                    z = reshape(sum(sum(dotproduct,1),3), [1, obj.numChannels]); % sum over kernel taps + input features
        
                    zs(i,:,k)    = z + obj.biases(:,k)';
                    a_out(i,:,k) = leakyReLU(zs(i,:,k),obj.a);
                end
            end

            % Cache for backprop
            if isTraining
                obj.actCache    = x_in;
                obj.preactCache = zs;
            end
        end

        %% Backprop: propagate gradients into input and update gradients
        function [d_in, dW_new, db_new] = backprop(obj, d_out)
            % Inputs:
            %   d_out  - backpropagated error at the output, [outputSteps × numChannels × outFeatures]
            % Outputs:
            %   d_in   - backpropagated error at the input,  [inputSteps × numChannels × inFeatures]
            %   dW_new - update vector of weights for the whole sequence
            %   db_new - update vector of biases for the whole sequence

            d_in     = zeros(size(obj.actCache));
            numSteps = size(obj.actCache,1)-obj.kernelSize+1;

            % accumulate new weight and bias grads
            dW_new   = zeros(size(obj.weights));
            db_new   = zeros(size(obj.biases));
        
            % Loop kernels and time for gradient computation
            for k = 1:obj.outFeatures
                for j = 1:obj.kernelSize
                    D = obj.actCache(j:end-obj.kernelSize+j,:,:) .* leakyReLU_prime(obj.preactCache(:,:,k),obj.a);
                    dW_new(j,:,k,:) = reshape(sum(D .* d_out(:,:,k),1), [1, obj.numChannels, 1, obj.inFeatures]);
                    if j == 1
                        Wk = reshape(obj.weights(:,:,k,:), [obj.kernelSize, obj.numChannels, obj.inFeatures]);
                        for i = 1:numSteps
                            gate = reshape(leakyReLU_prime(obj.preactCache(i,:,k),obj.a).*d_out(i,:,k), [1, obj.numChannels, 1]);
                            d_in(i:i+obj.kernelSize-1,:,:) = d_in(i:i+obj.kernelSize-1,:,:) + Wk .* gate;
                        end
                    end
                end
                db_new(:,k) = sum(d_out(:,:,k).*leakyReLU_prime(obj.preactCache(:,:,k),obj.a))';
            end
        end

        %% ApplyAdam: update weights and biases using Adam rule
        function applyAdam(obj, eta, beta1, beta2, m, t, eps)
            [obj.weights, obj.vdw, obj.sdw] = adamUpdate(obj.weights, obj.dW, obj.vdw, obj.sdw, beta1, beta2, t, eta, m, eps);
            [obj.biases,  obj.vdb, obj.sdb] = adamUpdate(obj.biases,  obj.db, obj.vdb, obj.sdb, beta1, beta2, t, eta, m, eps);
        end

        %% Resets
        function resetStoredActivations(obj)
            % Clear cached activations and pre-activations
            obj.actCache    = [];
            obj.preactCache = [];
        end

        function resetGrads(obj)
            % Zero gradients before accumulating new batch
            obj.dW = zeros(size(obj.weights));
            obj.db = zeros(size(obj.biases));
        end

    end
end

%% ReLU & ReLU'
function s = leakyReLU(z, a)
    s = max(a*z,z);
end

function s = leakyReLU_prime(z, a)
    s = ones(size(z));
    s(z < 0) = a;
end