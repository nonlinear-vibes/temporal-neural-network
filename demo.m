%% TemporalNeuralNet — demo
%   Synthesizes a 3-class time series (piecewise-constant labels) and simple features.
%   Builds and trains a small CNN → GRU → FC network.
%% add folders to path
if ~exist('GRUnit','file'), addpath('layers','utils'); end

%% 1) Make synthetic data
T = 600;        % timesteps
C = 4;          % channels
K = 3;          % classes

% Labels: first third - class 1, second third - class 2, third third - class 3
yIdx = ones(T,1);
yIdx((T/3+1):2*T/3) = 2;
yIdx((2*T/3+1):T)   = 3;
Y = onehot_encode(yIdx, K);      % [T x C] one-hot encoded

% Sequence: noise + class-specific shift
X = 0.5*randn(T, C);
X(1:T/3, 1:C)       = X(1:T/3, 1:C)       + 1;   % class 1
X(T/3+1:2*T/3, 1:C) = X(T/3+1:2*T/3, 1:C);       % class 2
X((2*T/3+1):T, 1:C) = X((2*T/3+1):T, 1:C) - 1;   % class 3

% Simple train/test split (two identical trials)
trainData = {X, Y; X, Y};
testData  = {X, Y};


%% 2) Build the network
% CNN
numKer = 1;
conv = { {'conv', C, 1, numKer, 3}, ...  % {'conv', inC, inLayers, numKernels, kernelSize}
        {'pool', 2} };                   % halving max-pool layer
 
% RNN (GRU) fed with flattened windows of length tPool
tPool = 2;
inDim = tPool * C * numKer;
rnn   = { {'gru', inDim, inDim} };    % GRU(inSize, outSize)

% Fully connected classifier to K classes
fc    = { [inDim, K] };

% Learning params
net = TemporalNeuralNet(testData, ...
    'CNN',conv, 'RNN',rnn, 'FC',fc, ...
    'tPool',tPool, 'numClasses',K, ...
    'eta',20, 'learningRateDecay',0.95);


%% 3) Train & evaluate
epochs    = 6;
batchSize = 20;
numSegs   = 10;     % per-trial base segments
                    % total segments per trial = numSegs + 3*(1-numSegs)

net.train(trainData, testData, epochs, batchSize, 'numSegments', numSegs);
acc = net.evaluate(testData);
fprintf('Validation accuracy: %.3f\n', acc);


%% 4) Visualize one run
probs = net.forward(X);          % [numSteps x K]
[~, pred] = max(probs, [], 2);

% Map step midpoints back to raw indices for display
midIdx = ((1:numel(pred))*floor(T/numel(pred)))-1;
figure('Name','TemporalNeuralNet demo'); 
subplot(2,1,1); hold on; grid on;
plot(midIdx, probs(:,1), 'LineWidth', 1.2);
plot(midIdx, probs(:,2), 'LineWidth', 1.2);
plot(midIdx, probs(:,3), 'LineWidth', 1.2);
ylim([0 1]); xlim([1 T]);
legend('P(class 1)','P(class 2)','P(class 2)','Location','best');
title('Per-step softmax probabilities');
xlabel('Time'); ylabel('Probability');
ylim('padded')

subplot(2,1,2); hold on; grid on;
plot(1:T, yIdx, 'k.');
stairs(midIdx, pred, 'LineWidth', 2);
xlim([1 T]); ylim([0.5 3.5]);
yticks([1 2 3]); yticklabels({'class 1','class 2','class 3'});
title('True vs predicted class');
xlabel('Time'); ylabel('Class');


%% Local helper
function Y = onehot_encode(idx, K)
% Convert integer labels (1..K) to one-hot matrix [N x K].
    N = numel(idx);
    Y = zeros(N, K);
    lin = sub2ind([N, K], (1:N)', idx(:));
    Y(lin) = 1;
end