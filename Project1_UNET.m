
clear; clc; close all;

load('Single_Cell.mat');   % Combined_Data: 281x1 cell of 80x80 double
load('Mask.mat');          % Combined_Mask: 281x1 cell of 80x80 logical

numSamples = numel(Combined_Data);
imgH = 80; imgW = 80;

% Allocate 4-D arrays: H x W x C x N (single channel grayscale)
allImages = zeros(imgH, imgW, 1, numSamples, 'single');
allMasks  = zeros(imgH, imgW, 1, numSamples, 'single');

for i = 1:numSamples
    img = single(Combined_Data{i});
    msk = single(Combined_Mask{i});
    % Normalize image to [0, 1]
    img = (img - min(img(:))) / (max(img(:)) - min(img(:)) + eps);
    allImages(:,:,1,i) = img;
    allMasks(:,:,1,i)  = msk;
end

% Train/Test split: first 200 for training, remaining 81 for testing
nTrain = 200;
nTest  = numSamples - nTrain;
trainImages = allImages(:,:,:,1:nTrain);
trainMasks  = allMasks(:,:,:,1:nTrain);
testImages  = allImages(:,:,:,nTrain+1:end);
testMasks   = allMasks(:,:,:,nTrain+1:end);

fprintf('Training samples: %d\n', size(trainImages, 4));
fprintf('Test samples:     %d\n', size(testImages, 4));
figure('Name', 'Sample Data', 'Position', [100 100 1200 400]);
for k = 1:5
    subplot(2, 5, k);
    imshow(trainImages(:,:,1,k), []);
    title(sprintf('Image %d', k));
    subplot(2, 5, k+5);
    imshow(trainMasks(:,:,1,k), []);
    title(sprintf('Mask %d', k));
end
sgtitle('Sample Training Image-Mask Pairs');

inputSize = [80 80 1];
numFilters = 16;

lgraph_baseline = buildUNet(inputSize, numFilters, 'withSkip', true, 'withBN', false);

opts_baseline = trainingOptions('adam', ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 16, ...
    'InitialLearnRate', 1e-3, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', true, ...
    'L2Regularization', 0);

fprintf('Training baseline U-Net (regression)...\n');
net_baseline = trainNetwork(trainImages, trainMasks, lgraph_baseline, opts_baseline);

% Predict on test set
predMasks_baseline = predict(net_baseline, testImages);
dice_baseline = computeMeanDice(predMasks_baseline, testMasks);
fprintf('Baseline U-Net (Regression) Mean Dice: %.4f\n', dice_baseline);

%% GUIDING QUESTION 1 - L2 REGULARIZATION SWEEP

fprintf('\n=== GUIDING QUESTION 1: L2 Regularization Sweep ===\n');

l2_values = [0, 1e-6, 1e-4, 1e-2];
dice_l2   = zeros(length(l2_values), 1);

for idx = 1:length(l2_values)
    fprintf('\n--- Training with L2 = %g ---\n', l2_values(idx));
    
    lgraph_l2 = buildUNet(inputSize, numFilters, 'withSkip', true, 'withBN', false);
    
    opts_l2 = trainingOptions('adam', ...
        'MaxEpochs', 50, ...
        'MiniBatchSize', 16, ...
        'InitialLearnRate', 1e-3, ...
        'Shuffle', 'every-epoch', ...
        'Plots', 'none', ...
        'Verbose', false, ...
        'L2Regularization', l2_values(idx));
    
    net_l2 = trainNetwork(trainImages, trainMasks, lgraph_l2, opts_l2);
    pred_l2 = predict(net_l2, testImages);
    dice_l2(idx) = computeMeanDice(pred_l2, testMasks);
    
    fprintf('L2 = %g  ->  Mean Dice = %.4f\n', l2_values(idx), dice_l2(idx));
end

% Plot Dice vs L2 factor
figure('Name', 'L2 Sweep', 'Position', [200 200 600 400]);
% Use a small value to represent 0 on log scale
xplot = l2_values;
xplot(1) = 1e-8;
semilogx(xplot, dice_l2, '-o', 'LineWidth', 2, 'MarkerSize', 8, ...
    'MarkerFaceColor', [0.2 0.4 0.8], 'Color', [0.2 0.4 0.8]);
xticks([1e-8 1e-6 1e-4 1e-2]);
xticklabels({'0', '1e-6', '1e-4', '1e-2'});
xlabel('L2 Regularization Factor', 'FontSize', 12);
ylabel('Mean Dice Score', 'FontSize', 12);
title('Dice Score vs. L2 Weight Decay Factor', 'FontSize', 14);
grid on;
set(gca, 'FontSize', 12);

[bestDice_l2, bestIdx_l2] = max(dice_l2);
fprintf('\n*** Best L2 = %g with Dice = %.4f ***\n', l2_values(bestIdx_l2), bestDice_l2);

%% GUIDING QUESTION 2 - WHY BCE > MSE FOR CLASSIFICATION

fprintf('\n=== GUIDING QUESTION 2: Cross-Entropy vs MSE Analysis ===\n');

% Numerical demonstration of gradient vanishing with MSE
fprintf('\n--- Gradient Magnitude Comparison (single pixel) ---\n');
fprintf('%-12s %-12s %-18s %-18s\n', 'p (pred)', 'y (true)', 'dL/dz (MSE)', 'dL/dz (BCE)');
fprintf('%s\n', repmat('-', 1, 60));

y = 1;  % ground truth = cell
p_values = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99];
for p = p_values
    grad_mse = 2 * (p - y) * p * (1 - p);   % MSE gradient through sigmoid
    grad_bce = p - y;                         % BCE gradient (sigmoid cancels)
    fprintf('%-12.2f %-12d %-18.6f %-18.6f\n', p, y, abs(grad_mse), abs(grad_bce));
end

fprintf('\nKey insight: When p=0.01 (confident wrong prediction):\n');
fprintf('  MSE gradient = %.6f  (nearly zero -> stuck!)\n', abs(2*(0.01-1)*0.01*0.99));
fprintf('  BCE gradient = %.6f  (strong signal -> fast correction)\n', abs(0.01-1));
fprintf('  BCE gradient is %.0fx larger!\n', abs(0.01-1) / abs(2*(0.01-1)*0.01*0.99));

%% GUIDING QUESTION 3 - SKIP CONNECTION ABLATION
fprintf('\n=== GUIDING QUESTION 3: Skip Connection Ablation ===\n');

lgraph_noskip = buildUNet(inputSize, numFilters, 'withSkip', false, 'withBN', false);

opts_noskip = trainingOptions('adam', ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 16, ...
    'InitialLearnRate', 1e-3, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'none', ...
    'Verbose', true);

fprintf('Training U-Net WITHOUT skip connections...\n');
net_noskip = trainNetwork(trainImages, trainMasks, lgraph_noskip, opts_noskip);

pred_noskip = predict(net_noskip, testImages);
dice_noskip = computeMeanDice(pred_noskip, testMasks);

fprintf('\nU-Net WITH skip connections:    Mean Dice = %.4f\n', dice_baseline);
fprintf('U-Net WITHOUT skip connections: Mean Dice = %.4f\n', dice_noskip);
fprintf('Performance drop: %.2f%%\n', 100*(dice_baseline - dice_noskip) / (dice_baseline + eps));

% Visualize where degradation occurs
figure('Name', 'Skip Connection Ablation', 'Position', [100 100 1400 600]);
for k = 1:4
    err_skip   = abs(double(predMasks_baseline(:,:,1,k)) - double(testMasks(:,:,1,k)));
    err_noskip = abs(double(pred_noskip(:,:,1,k)) - double(testMasks(:,:,1,k)));
    diff_err   = err_noskip - err_skip;  % positive = worse without skip
    
    % Row 1: Extra error heatmap
    subplot(3, 4, k);
    imagesc(diff_err); colormap(gca, 'hot'); colorbar;
    title(sprintf('Extra Error #%d', k)); axis image off;
    
    % Row 2: Degradation overlay on image
    subplot(3, 4, k+4);
    img_rgb = repmat(testImages(:,:,1,k), [1 1 3]);
    red_overlay = cat(3, ones(80,'single'), zeros(80,'single'), zeros(80,'single'));
    alpha = 0.5 * single(max(diff_err, 0));
    combined = img_rgb .* (1-alpha) + red_overlay .* alpha;
    imshow(combined);
    title(sprintf('Degradation Overlay #%d', k));
    
    % Row 3: Side-by-side predicted masks
    subplot(3, 4, k+8);
    pred_with    = predMasks_baseline(:,:,1,k) > 0.5;
    pred_without = pred_noskip(:,:,1,k) > 0.5;
    gt           = testMasks(:,:,1,k) > 0.5;
    comparison   = cat(3, single(pred_without), single(pred_with), single(gt));
    imshow(comparison);
    title('R=NoSkip G=Skip B=GT');
end
sgtitle('Skip Connection Ablation: Spatial Degradation Analysis');

%%  GUIDING QUESTION 4 - BATCH NORMALIZATION
fprintf('\n=== GUIDING QUESTION 4: Batch Normalization Comparison ===\n');

% We'll capture loss histories using OutputFcn
lossHistory = struct('noBN', [], 'withBN', []);

% Re-train baseline (no BN) to capture loss curve
lgraph_noBN = buildUNet(inputSize, numFilters, 'withSkip', true, 'withBN', false);
opts_noBN = trainingOptions('adam', ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 16, ...
    'InitialLearnRate', 1e-3, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'none', ...
    'Verbose', false, ...
    'OutputFcn', @(info) collectLoss(info, 'noBN'));

fprintf('Training U-Net WITHOUT batch normalization...\n');
net_noBN = trainNetwork(trainImages, trainMasks, lgraph_noBN, opts_noBN);

% Train with BN
lgraph_withBN = buildUNet(inputSize, numFilters, 'withSkip', true, 'withBN', true);
opts_withBN = trainingOptions('adam', ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 16, ...
    'InitialLearnRate', 1e-3, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'none', ...
    'Verbose', false, ...
    'OutputFcn', @(info) collectLoss(info, 'withBN'));

fprintf('Training U-Net WITH batch normalization...\n');
net_withBN = trainNetwork(trainImages, trainMasks, lgraph_withBN, opts_withBN);

% Evaluate both
pred_noBN  = predict(net_noBN, testImages);
pred_withBN = predict(net_withBN, testImages);
dice_noBN  = computeMeanDice(pred_noBN, testMasks);
dice_withBN = computeMeanDice(pred_withBN, testMasks);

fprintf('\nWithout BN: Mean Dice = %.4f\n', dice_noBN);
fprintf('With BN:    Mean Dice = %.4f\n', dice_withBN);

% Retrieve loss histories from appdata
loss_noBN  = getappdata(0, 'loss_noBN');
loss_withBN = getappdata(0, 'loss_withBN');

% Plot comparison
figure('Name', 'Batch Normalization', 'Position', [200 200 900 400]);
subplot(1,2,1);
plot(loss_noBN, 'b-', 'LineWidth', 1, 'DisplayName', 'Without BN'); hold on;
plot(loss_withBN, 'r-', 'LineWidth', 1, 'DisplayName', 'With BN');
xlabel('Iteration', 'FontSize', 12); ylabel('Training Loss (MSE)', 'FontSize', 12);
title('Training Loss Curves', 'FontSize', 14);
legend('Location', 'northeast', 'FontSize', 11); grid on;

subplot(1,2,2);
bar([dice_noBN, dice_withBN], 0.5);
set(gca, 'XTickLabel', {'Without BN', 'With BN'});
ylabel('Mean Dice Score', 'FontSize', 12);
title('Test Performance', 'FontSize', 14);
grid on; ylim([0 1]);
sgtitle('Effect of Batch Normalization');

% Estimate convergence epoch
smoothed_noBN  = movmean(loss_noBN, 50);
smoothed_withBN = movmean(loss_withBN, 50);
threshold = min(smoothed_noBN(end), smoothed_withBN(end)) * 1.1;
itersPerEpoch = ceil(nTrain / 16);
conv_noBN  = find(smoothed_noBN < threshold, 1, 'first');
conv_withBN = find(smoothed_withBN < threshold, 1, 'first');
if ~isempty(conv_noBN) && ~isempty(conv_withBN)
    fprintf('\nApprox convergence (within 10%% of final loss):\n');
    fprintf('  Without BN: epoch ~%d (iter %d)\n', ceil(conv_noBN/itersPerEpoch), conv_noBN);
    fprintf('  With BN:    epoch ~%d (iter %d)\n', ceil(conv_withBN/itersPerEpoch), conv_withBN);
end


%%  To Do List 3 - SGD vs ADAM OPTIMIZER COMPARISON
fprintf('\n=== To Do List 3: SGD vs Adam ===\n');

% Reset loss storage
setappdata(0, 'loss_sgd', []);
setappdata(0, 'loss_adam', []);

lgraph_sgd  = buildUNet(inputSize, numFilters, 'withSkip', true, 'withBN', true);
lgraph_adam2 = buildUNet(inputSize, numFilters, 'withSkip', true, 'withBN', true);

opts_sgd = trainingOptions('sgdm', ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 16, ...
    'InitialLearnRate', 1e-4, ...
    'Momentum', 0.9, ...
    'GradientThreshold', 1, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'none', ...
    'Verbose', false, ...
    'OutputFcn', @(info) collectLoss(info, 'sgd'));

opts_adam2 = trainingOptions('adam', ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 16, ...
    'InitialLearnRate', 1e-3, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'none', ...
    'Verbose', false, ...
    'OutputFcn', @(info) collectLoss(info, 'adam'));

fprintf('Training with SGD...\n');
net_sgd = trainNetwork(trainImages, trainMasks, lgraph_sgd, opts_sgd);
fprintf('Training with Adam...\n');
net_adam2 = trainNetwork(trainImages, trainMasks, lgraph_adam2, opts_adam2);

pred_sgd  = predict(net_sgd, testImages);
pred_adam2 = predict(net_adam2, testImages);
dice_sgd  = computeMeanDice(pred_sgd, testMasks);
dice_adam2 = computeMeanDice(pred_adam2, testMasks);

fprintf('\nSGD  Mean Dice: %.4f\n', dice_sgd);
fprintf('Adam Mean Dice: %.4f\n', dice_adam2);

% Retrieve losses
loss_sgd  = getappdata(0, 'loss_sgd');
loss_adam2 = getappdata(0, 'loss_adam');

figure('Name', 'SGD vs Adam', 'Position', [200 200 900 400]);
subplot(1,2,1);
plot(loss_sgd, 'b-', 'LineWidth', 1, 'DisplayName', 'SGD'); hold on;
plot(loss_adam2, 'r-', 'LineWidth', 1, 'DisplayName', 'Adam');
xlabel('Iteration'); ylabel('Training Loss');
title('Training Loss: SGD vs Adam'); legend; grid on;

subplot(1,2,2);
b = bar([dice_sgd, dice_adam2], 0.5);
b.FaceColor = 'flat';
b.CData = [0.2 0.4 0.8; 0.8 0.3 0.2];
set(gca, 'XTickLabel', {'SGD', 'Adam'});
ylabel('Mean Dice Score'); title('Test Performance'); grid on; ylim([0 1]);


%% To Do List 4 - EVALUATION METRICS - DICE vs IoU vs BCE
fprintf('\n=== To Do List 4: Evaluation Metrics Comparison ===\n');

diceScores = zeros(nTest, 1);
bceScores  = zeros(nTest, 1);

for i = 1:nTest
    p = double(pred_adam2(:,:,1,i));
    g = double(testMasks(:,:,1,i));
    
    p_bin = p > 0.5;
    g_bin = g > 0.5;

    intersection = sum(p_bin(:) & g_bin(:));
    diceScores(i) = (2 * intersection) / (sum(p_bin(:)) + sum(g_bin(:)) + eps);
    
    p_clip = max(min(p, 1 - 1e-7), 1e-7);
    bce = -(g .* log(p_clip) + (1 - g) .* log(1 - p_clip));
    bceScores(i) = mean(bce(:));
end

fprintf('Mean Dice: %.4f +/- %.4f\n', mean(diceScores), std(diceScores));
fprintf('Mean BCE:  %.4f +/- %.4f\n', mean(bceScores),  std(bceScores));

figure('Name', 'Metrics Comparison: BCE vs Dice', 'Position', [200 200 600 500]);
scatter(bceScores, diceScores, 40, 'filled', 'MarkerFaceColor', [0.2 0.5 0.8]);
hold on;

pfit = polyfit(bceScores, diceScores, 1);
xfit = linspace(min(bceScores), max(bceScores), 100);
yfit = polyval(pfit, xfit);
plot(xfit, yfit, 'r-', 'LineWidth', 2, 'DisplayName', 'Trendline');
hold off;

xlabel('BCE Loss ', 'FontSize', 12); 
ylabel('Dice Score ', 'FontSize', 12);
title('Per-Sample Comparison: BCE vs Dice', 'FontSize', 14); 
grid on;

%% To Do List 5 - PREDICTED MASK OVERLAY VISUALIZATION
fprintf('\n=== SECTION 10: Predicted Mask Overlay ===\n');

nShow = min(6, nTest);
figure('Name', 'Predicted Mask Overlay', 'Position', [50 50 1500 700]);
for k = 1:nShow
    % Row 1: Original image
    subplot(3, nShow, k);
    imshow(testImages(:,:,1,k), []);
    title(sprintf('Test #%d', k));
    
    % Row 2: Predicted mask
    subplot(3, nShow, k + nShow);
    pred_bin = pred_adam2(:,:,1,k) > 0.5;
    imshow(pred_bin, []);
    title('Predicted Mask');
    
    % Row 3: Color-coded overlay (Green=TP, Red=FP, Blue=FN)
    subplot(3, nShow, k + 2*nShow);
    img_rgb = repmat(testImages(:,:,1,k), [1 1 3]);
    gt_bin  = testMasks(:,:,1,k) > 0.5;
    
    tp = single(pred_bin & gt_bin);    % true positive  -> green
    fp = single(pred_bin & ~gt_bin);   % false positive -> red
    fn = single(~pred_bin & gt_bin);   % false negative -> blue
    
    overlay = img_rgb;
    overlay(:,:,1) = min(overlay(:,:,1) + 0.4*fp, 1);
    overlay(:,:,2) = min(overlay(:,:,2) + 0.4*tp, 1);
    overlay(:,:,3) = min(overlay(:,:,3) + 0.4*fn, 1);
    imshow(overlay);
    title('G=TP R=FP B=FN');
end
sgtitle('Test Set Predictions with Error Overlay', 'FontSize', 14);


function lgraph = buildUNet(inputSize, nf, opts)
    arguments
        inputSize (1,3) double
        nf (1,1) double = 16
        opts.withSkip (1,1) logical = true
        opts.withBN   (1,1) logical = false
    end
    
    useSkip = opts.withSkip;
    useBN   = opts.withBN;
    
    %=Encoder 1=
    enc1 = [
        imageInputLayer(inputSize, 'Name', 'input', 'Normalization', 'none')
        convolution2dLayer(3, nf, 'Padding', 'same', 'Name', 'enc1_conv1')
    ];
    if useBN, enc1 = [enc1; batchNormalizationLayer('Name','enc1_bn1')]; end
    enc1 = [enc1; reluLayer('Name','enc1_relu1')];
    enc1 = [enc1; convolution2dLayer(3, nf, 'Padding', 'same', 'Name', 'enc1_conv2')];
    if useBN, enc1 = [enc1; batchNormalizationLayer('Name','enc1_bn2')]; end
    enc1 = [enc1; reluLayer('Name','enc1_relu2')];
    enc1 = [enc1; maxPooling2dLayer(2, 'Stride', 2, 'Name', 'enc1_pool')]; 
    
    % Encoder 2
    enc2 = [
        convolution2dLayer(3, nf*2, 'Padding', 'same', 'Name', 'enc2_conv1')
    ];
    if useBN, enc2 = [enc2; batchNormalizationLayer('Name','enc2_bn1')]; end
    enc2 = [enc2; reluLayer('Name','enc2_relu1')];
    enc2 = [enc2; convolution2dLayer(3, nf*2, 'Padding', 'same', 'Name', 'enc2_conv2')];
    if useBN, enc2 = [enc2; batchNormalizationLayer('Name','enc2_bn2')]; end
    enc2 = [enc2; reluLayer('Name','enc2_relu2')];
    
    bn = [
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'enc2_pool')
        convolution2dLayer(3, nf*4, 'Padding', 'same', 'Name', 'bn_conv1')
    ];
    if useBN, bn = [bn; batchNormalizationLayer('Name','bn_bn1')]; end
    bn = [bn; reluLayer('Name','bn_relu1')];
    bn = [bn; convolution2dLayer(3, nf*4, 'Padding', 'same', 'Name', 'bn_conv2')];
    if useBN, bn = [bn; batchNormalizationLayer('Name','bn_bn2')]; end
    bn = [bn; reluLayer('Name','bn_relu2')];
    
    if useSkip
        % ---- With skip connections ----
        lgraph = layerGraph([enc1; enc2; bn]);
        
        % Decoder Level 2
        lgraph = addLayers(lgraph, transposedConv2dLayer(2, nf*2, 'Stride', 2, 'Name', 'upconv2'));
        lgraph = addLayers(lgraph, reluLayer('Name', 'upconv2_relu'));
        
        dec2 = [depthConcatenationLayer(2, 'Name', 'concat2')
                convolution2dLayer(3, nf*2, 'Padding','same','Name','dec2_conv1')];
        if useBN, dec2 = [dec2; batchNormalizationLayer('Name','dec2_bn1')]; end
        dec2 = [dec2; reluLayer('Name','dec2_relu1')];
        dec2 = [dec2; convolution2dLayer(3, nf*2, 'Padding','same','Name','dec2_conv2')];
        if useBN, dec2 = [dec2; batchNormalizationLayer('Name','dec2_bn2')]; end
        dec2 = [dec2; reluLayer('Name','dec2_relu2')];
        lgraph = addLayers(lgraph, dec2);
        
        % Decoder Level 1
        lgraph = addLayers(lgraph, transposedConv2dLayer(2, nf, 'Stride', 2, 'Name', 'upconv1'));
        lgraph = addLayers(lgraph, reluLayer('Name', 'upconv1_relu')); 
        
        dec1 = [depthConcatenationLayer(2, 'Name', 'concat1')
                convolution2dLayer(3, nf, 'Padding','same','Name','dec1_conv1')];
        if useBN, dec1 = [dec1; batchNormalizationLayer('Name','dec1_bn1')]; end
        dec1 = [dec1; reluLayer('Name','dec1_relu1')];
        dec1 = [dec1; convolution2dLayer(3, nf, 'Padding','same','Name','dec1_conv2')];
        if useBN, dec1 = [dec1; batchNormalizationLayer('Name','dec1_bn2')]; end
        dec1 = [dec1; reluLayer('Name','dec1_relu2')];
        lgraph = addLayers(lgraph, dec1);
        
        % Output
        lgraph = addLayers(lgraph, [
            convolution2dLayer(1, 1, 'Name', 'final_conv')
            regressionLayer('Name', 'output')
        ]);
        
        % Connections
        lgraph = connectLayers(lgraph, 'bn_relu2',  'upconv2');
        lgraph = connectLayers(lgraph, 'upconv2', 'upconv2_relu');
        lgraph = connectLayers(lgraph, 'upconv2_relu',   'concat2/in1');
        lgraph = connectLayers(lgraph, 'enc2_relu2',  'concat2/in2');   % skip 2
        lgraph = connectLayers(lgraph, 'dec2_relu2',  'upconv1');
        lgraph = connectLayers(lgraph, 'upconv1', 'upconv1_relu');
        lgraph = connectLayers(lgraph, 'upconv1_relu',   'concat1/in1');
        lgraph = connectLayers(lgraph, 'enc1_relu2',  'concat1/in2');   % skip 1
        lgraph = connectLayers(lgraph, 'dec1_relu2',  'final_conv');
        
    else
        dec2_noskip = [
            transposedConv2dLayer(2, nf*2, 'Stride', 2, 'Name', 'upconv2')
            reluLayer('Name', 'upconv2_relu')
            convolution2dLayer(3, nf*2, 'Padding','same','Name','dec2_conv1')
        ];
        if useBN, dec2_noskip = [dec2_noskip; batchNormalizationLayer('Name','dec2_bn1')]; end
        dec2_noskip = [dec2_noskip; reluLayer('Name','dec2_relu1')];
        dec2_noskip = [dec2_noskip; convolution2dLayer(3, nf*2, 'Padding','same','Name','dec2_conv2')];
        if useBN, dec2_noskip = [dec2_noskip; batchNormalizationLayer('Name','dec2_bn2')]; end
        dec2_noskip = [dec2_noskip; reluLayer('Name','dec2_relu2')];
        
        dec1_noskip = [
            transposedConv2dLayer(2, nf, 'Stride', 2, 'Name', 'upconv1')
            reluLayer('Name', 'upconv1_relu')
            convolution2dLayer(3, nf, 'Padding','same','Name','dec1_conv1')
        ];
        if useBN, dec1_noskip = [dec1_noskip; batchNormalizationLayer('Name','dec1_bn1')]; end
        dec1_noskip = [dec1_noskip; reluLayer('Name','dec1_relu1')];
        dec1_noskip = [dec1_noskip; convolution2dLayer(3, nf, 'Padding','same','Name','dec1_conv2')];
        if useBN, dec1_noskip = [dec1_noskip; batchNormalizationLayer('Name','dec1_bn2')]; end
        dec1_noskip = [dec1_noskip; reluLayer('Name','dec1_relu2')];
        
        allLayers = [enc1; enc2; bn; dec2_noskip; dec1_noskip;
            convolution2dLayer(1, 1, 'Name', 'final_conv')
            regressionLayer('Name', 'output')
        ];
        lgraph = layerGraph(allLayers);
    end
end



%% computeMeanDice: Average Dice coefficient over a batch
function meanDice = computeMeanDice(pred, gt)
    % Computes mean Dice score across all samples in the batch.
    % pred, gt: H x W x 1 x N arrays (values in [0,1])
    
    pred_bin = pred > 0.5;
    gt_bin   = gt > 0.5;
    N = size(pred, 4);
    diceScores = zeros(N, 1);
    
    for i = 1:N
        p = pred_bin(:,:,1,i);
        g = gt_bin(:,:,1,i);
        intersection = sum(p(:) & g(:));
        diceScores(i) = (2 * intersection) / (sum(p(:)) + sum(g(:)) + eps);
    end
    
    meanDice = mean(diceScores);
end

%% collectLoss: OutputFcn callback for recording training loss
function stop = collectLoss(info, tag)
    % Stores training loss values in appdata for later plotting.
    % Usage: 'OutputFcn', @(info) collectLoss(info, 'tagName')
    
    stop = false;
    key = ['loss_' tag];
    
    if ~isempty(info.TrainingLoss)
        existing = getappdata(0, key);
        if isempty(existing)
            existing = [];
        end
        existing(end+1) = info.TrainingLoss; 
        setappdata(0, key, existing);
    end
end

%% buildClassificationUNet: 2-layer U-Net for CLASSIFICATION (BCE loss)
function lgraph = buildClassificationUNet(inputSize, nf, classNames, classWeights)
    % Encoder 1
    enc1 = [
        imageInputLayer(inputSize, 'Name', 'input', 'Normalization', 'none')
        convolution2dLayer(3, nf, 'Padding', 'same', 'Name', 'enc1_conv1')
        batchNormalizationLayer('Name','enc1_bn1')
        reluLayer('Name','enc1_relu1')
        convolution2dLayer(3, nf, 'Padding', 'same', 'Name', 'enc1_conv2')
        batchNormalizationLayer('Name','enc1_bn2')
        reluLayer('Name','enc1_relu2')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'enc1_pool')
    ];
    
    % Encoder 2
    enc2 = [
        convolution2dLayer(3, nf*2, 'Padding', 'same', 'Name', 'enc2_conv1')
        batchNormalizationLayer('Name','enc2_bn1')
        reluLayer('Name','enc2_relu1')
        convolution2dLayer(3, nf*2, 'Padding', 'same', 'Name', 'enc2_conv2')
        batchNormalizationLayer('Name','enc2_bn2')
        reluLayer('Name','enc2_relu2')
    ];
    
    % Bottleneck 
    bn = [
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'enc2_pool')
        convolution2dLayer(3, nf*4, 'Padding', 'same', 'Name', 'bn_conv1')
        batchNormalizationLayer('Name','bn_bn1')
        reluLayer('Name','bn_relu1')
        convolution2dLayer(3, nf*4, 'Padding', 'same', 'Name', 'bn_conv2')
        batchNormalizationLayer('Name','bn_bn2')
        reluLayer('Name','bn_relu2')
    ];
    
    lgraph = layerGraph([enc1; enc2; bn]);
    
    % ---- Decoder Level 2 ----
    lgraph = addLayers(lgraph, transposedConv2dLayer(2, nf*2, 'Stride', 2, 'Name', 'upconv2'));
    lgraph = addLayers(lgraph, reluLayer('Name', 'upconv2_relu')); 
    
    dec2 = [
        depthConcatenationLayer(2, 'Name', 'concat2')
        convolution2dLayer(3, nf*2, 'Padding','same','Name','dec2_conv1')
        batchNormalizationLayer('Name','dec2_bn1')
        reluLayer('Name','dec2_relu1')
        convolution2dLayer(3, nf*2, 'Padding','same','Name','dec2_conv2')
        batchNormalizationLayer('Name','dec2_bn2')
        reluLayer('Name','dec2_relu2')
    ];
    lgraph = addLayers(lgraph, dec2);
    
    % ---- Decoder Level 1 ----
    lgraph = addLayers(lgraph, transposedConv2dLayer(2, nf, 'Stride', 2, 'Name', 'upconv1'));
    lgraph = addLayers(lgraph, reluLayer('Name', 'upconv1_relu')); 
    
    dec1 = [
        depthConcatenationLayer(2, 'Name', 'concat1')
        convolution2dLayer(3, nf, 'Padding','same','Name','dec1_conv1')
        batchNormalizationLayer('Name','dec1_bn1')
        reluLayer('Name','dec1_relu1')
        convolution2dLayer(3, nf, 'Padding','same','Name','dec1_conv2')
        batchNormalizationLayer('Name','dec1_bn2')
        reluLayer('Name','dec1_relu2')
    ];
    lgraph = addLayers(lgraph, dec1);
    outLayers = [
        convolution2dLayer(1, 2, 'Name', 'final_conv')
        softmaxLayer('Name', 'softmax')
        pixelClassificationLayer('Name', 'output', 'Classes', classNames, 'ClassWeights', classWeights)
    ];
    lgraph = addLayers(lgraph, outLayers);
    
    lgraph = connectLayers(lgraph, 'bn_relu2',  'upconv2');
    lgraph = connectLayers(lgraph, 'upconv2', 'upconv2_relu'); 
    lgraph = connectLayers(lgraph, 'upconv2_relu',   'concat2/in1');
    lgraph = connectLayers(lgraph, 'enc2_relu2',  'concat2/in2');   % skip 2
    
    lgraph = connectLayers(lgraph, 'dec2_relu2',  'upconv1');
    lgraph = connectLayers(lgraph, 'upconv1', 'upconv1_relu'); 
    lgraph = connectLayers(lgraph, 'upconv1_relu',   'concat1/in1');
    lgraph = connectLayers(lgraph, 'enc1_relu2',  'concat1/in2');   % skip 1
    
    lgraph = connectLayers(lgraph, 'dec1_relu2',  'final_conv');
end
