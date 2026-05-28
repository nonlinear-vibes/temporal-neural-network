classdef ConvolutionalLayer < handle
    %% ConvolutionalLayer: separate kernels scanning through each channel
    properties
        numChannels
        inFeatures
        outFeatures
        kernelSize
        weights
        biases
        acts
        preacts
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
            meanWeights = squeeze(mean(obj.weights,1));
            biasValues  = mean(meanWeights,3);

            if size(biasValues,1) == 1
                obj.biases = -biasValues'; 
            else
                obj.biases = -biasValues;   
            end

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
        function as = forward(obj, input, varargin)
            % input: [T×numChannels×inFeatures]

            doCache  = ~isempty(varargin) && strcmp(varargin{1}, 'train');

            numSteps = size(input,1) - obj.kernelSize + 1;

            as = zeros(numSteps, obj.numChannels, obj.outFeatures);  % activations
            zs = zeros(numSteps, obj.numChannels, obj.outFeatures);  % pre-activations

            % Perform convolution
            for i = 1:numSteps
                step_Index = 1 + (i-1);
                for k = 1:obj.outFeatures
                    dotproduct = input(step_Index:step_Index+obj.kernelSize-1,:,:) .* squeeze(obj.weights(:,:,k,:));
                    z = sum(dotproduct);
                    if length(size(z)) ~= 2
                        z = sum(z,4);
                        z = reshape(z,[obj.numChannels,obj.inFeatures])';
                        if size(z,1) ~= 1
                            z = sum(z);
                        end
                    end
                    zs(i,:,k) = z + obj.biases(:,k)';
                    as(i,:,k) = leakyReLU(zs(i,:,k),obj.a);
                end
            end

            % Cache for backprop
            if doCache
                obj.acts    = input;
                obj.preacts = zs;
            end
        end

        %% Backprop: propagate gradients into input and update gradients
        function [d_input, dW_new, db_new] = backprop(obj, d_output)
            % d_output: [T×numChannels×outFeatures]

            d_input  = zeros(size(obj.acts));
            numSteps = size(obj.acts,1)-obj.kernelSize+1;

            % accumulate new weight and bias grads
            dW_new   = zeros(size(obj.weights));
            db_new   = zeros(size(obj.biases));
        
            % Loop kernels and time for gradient computation
            for k = 1:obj.outFeatures
                for j = 1:obj.kernelSize
                    D = obj.acts(j:end-obj.kernelSize+j,:,:) .* leakyReLU_prime(obj.preacts(:,:,k),obj.a);
                    dW_new(j,:,k,:) = squeeze(sum(D .* d_output(:,:,k),1));
                    if j == 1
                        for i = 1:numSteps
                            d_input(i:i+obj.kernelSize-1,:,:) = d_input(i:i+obj.kernelSize-1,:,:) + squeeze(obj.weights(:,:,k,:)).*reshape(leakyReLU_prime(obj.preacts(i,:,k),obj.a).*d_output(i,:,k),[1,obj.numChannels,1]);
                        end
                    end
                end
                db_new(:,k) = sum(d_output(:,:,k).*leakyReLU_prime(obj.preacts(:,:,k),obj.a))';
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
            obj.acts    = [];
            obj.preacts = [];
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