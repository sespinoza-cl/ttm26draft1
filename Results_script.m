%REPRODUCIBILITY SCRIPT: TMD PHENOTYPES & CERVICAL COMORBIDITY
% Author: Sebastian Espinoza, PhD.
% Description: Reproduces the PCA, K-means clustering, and robust logistic 
% regression analysis for the paper: "Masticatory nociceptive burden predicts 
% cervical comorbidity in patients with temporomandibular disorders"

clear all; clc; close all;

%% 1. DATA LOADING
% Assumes 'TMD_Cervical_Dataset.txt' is in the current working directory
fileName = 'TMD_Cervical_Dataset.txt';
if ~exist(fileName, 'file')
    error('Dataset not found. Please ensure %s is in the MATLAB path.', fileName);
end

% Robust import: Read the tab-separated TXT file
data = readtable(fileName, 'Delimiter', '\t', 'ReadVariableNames', true, 'VariableNamingRule', 'preserve');

fprintf('Dataset loaded successfully: n=%d patients.\n', height(data));

%% 2. MULTIVARIATE ANALYSIS: PCA & K-MEANS
% Extract diagnostic variables robustly (forcing numeric column vectors)
try
    % Converting to string then double ensures safety just in case readtable infers cells
    m_pain = double(string(data.Muscle_Pain(:)));
    j_pain = double(string(data.Joint_Pain(:)));
    i_disorder = double(string(data.Intra_Articular_Disorder(:)));
catch ME
    disp('Error reading columns. Current table headers are:');
    disp(data.Properties.VariableNames);
    rethrow(ME);
end

X_raw = [m_pain, j_pain, i_disorder];
X_scaled = zscore(X_raw); % Standardize for PCA and Clustering

% Perform PCA
[coeff, score, ~, ~, explained] = pca(X_scaled);

% K-Means Clustering (k=3)
rng(42); % Fixed seed for reproducibility
[idx, C] = kmeans(X_scaled, 3, 'Replicates', 50);
data.Cluster = idx;

% Dynamic Phenotype Naming based on cluster centroids
phenoNames = cell(1,3);
for i = 1:3
    m = mean(m_pain(idx==i)); 
    j = mean(j_pain(idx==i));
    if m > 0.8 && j > 0.8
        phenoNames{i} = 'Nociceptive';
    elseif m < 0.2 && j < 0.2
        phenoNames{i} = 'Structural';
    else
        phenoNames{i} = 'Myogenic';
    end
end

%% 3. FIGURE 1: SILHOUETTE VALIDATION
f1 = figure('Color', 'w', 'Units', 'inches', 'Position', [1 1 8 6]);
s_vals_raw = silhouette(X_scaled, idx, 'Euclidean');

% Sort for solid block visualization
[sorted_data, sort_idx] = sortrows([idx, s_vals_raw], [1, -2]);
s_vals = sorted_data(:,2);
c_vals = sorted_data(:,1);

% Nature-inspired Palette: Nociceptive, Structural, Myogenic
nature_palette = [0.46, 0.71, 0.63; 0.50, 0.69, 0.82; 0.91, 0.59, 0.58]; 

hold on;
tick_pos = zeros(1,3);
for i = 1:3
    cluster_mask = (c_vals == i);
    y_indices = find(cluster_mask);
    barh(y_indices, s_vals(cluster_mask), 1.0, 'FaceColor', nature_palette(i,:), 'EdgeColor', 'none');
    tick_pos(i) = mean(y_indices);
end

set(gca, 'YTick', tick_pos, 'YTickLabel', phenoNames, 'FontSize', 14, 'FontWeight', 'bold', 'TickDir', 'out', 'LineWidth', 1.5);
xlabel('Silhouette Coefficient', 'FontSize', 16, 'FontWeight', 'bold');
title('Phenotype Cluster Validation', 'FontSize', 18, 'FontWeight', 'bold');
box off; a = gca; a.Toolbar.Visible = 'off';

%% 4. FIGURE 2: PCA MAPPING
f2 = figure('Color', 'w', 'Units', 'inches', 'Position', [1 1 8 7]);
h_scat = gscatter(score(:,1), score(:,2), data.Cluster, nature_palette, 'osd', 14);

for i = 1:numel(h_scat)
    set(h_scat(i), 'MarkerFaceColor', get(h_scat(i), 'Color'), 'MarkerEdgeColor', 'w', 'LineWidth', 1.5);
end

xlabel(['PC1 (Nociceptive Burden) - ', num2str(explained(1),1), '%'], 'FontSize', 16, 'FontWeight', 'bold');
ylabel(['PC2 (Structural Factor) - ', num2str(explained(2),1), '%'], 'FontSize', 16, 'FontWeight', 'bold');
title('TMD Phenotype Distribution in PCA Space', 'FontSize', 18, 'FontWeight', 'bold');
legend(phenoNames, 'Location', 'best', 'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'FontSize', 15, 'FontWeight', 'bold', 'TickDir', 'out', 'LineWidth', 1.5);
box off; a = gca; a.Toolbar.Visible = 'off';

%% 5. MULTIVARIATE LOGISTIC REGRESSION
fprintf('\n--- MULTIVARIATE LOGISTIC REGRESSION ---\n');
% Safely extract and format variables for the model
target_y = double(string(data.Neck_Pain(:)));
pred_age = double(string(data.Age(:)));
pred_sex = strcmpi(string(data.Sex(:)), 'Female');

% Build table for model
tbl_model = table(target_y, score(:,1), score(:,2), pred_sex, pred_age, ...
    'VariableNames', {'NeckPain','PC1','PC2','Sex_Female','Age'});

mdl = fitglm(tbl_model, 'NeckPain ~ PC1 + PC2 + Sex_Female + Age', 'Distribution', 'binomial');

% Display standard results
OR = exp(mdl.Coefficients.Estimate); 
SE = mdl.Coefficients.SE;
ResTable = table(mdl.CoefficientNames', OR, exp(mdl.Coefficients.Estimate - 1.96*SE), ...
    exp(mdl.Coefficients.Estimate + 1.96*SE), mdl.Coefficients.pValue, ...
    'VariableNames', {'Variable','OddsRatio','LowCI95','UpCI95','PValue'});
disp(ResTable);

%% 6. ROBUST BOOTSTRAP ANALYSIS (PC1)
fprintf('\n--- RUNNING BOOTSTRAP ANALYSIS (n=2000) ---\n');
nBoot = 2000;
bootCoefs = zeros(nBoot, length(mdl.CoefficientNames));
rng(42); % Fixed seed for exact reproducibility

for i = 1:nBoot
    idx_boot = randsample(height(tbl_model), height(tbl_model), true);
    tbl_boot = tbl_model(idx_boot, :);
    
    try
        mdl_boot = fitglm(tbl_boot, 'NeckPain ~ PC1 + PC2 + Sex_Female + Age', ...
            'Distribution', 'binomial', 'Link', 'logit');
        bootCoefs(i, :) = mdl_boot.Coefficients.Estimate;
    catch
        continue; 
    end
end

% Calculate Empirical Stats for PC1 (Index 2)
bootOR_PC1 = exp(mean(bootCoefs(:, 2))); 
p_boot_pc1 = mean(bootCoefs(:, 2) <= 0) * 2; 
ci_boot_pc1 = [exp(quantile(bootCoefs(:, 2), 0.025)), exp(quantile(bootCoefs(:, 2), 0.975))];

fprintf('Variable: PC1 (Nociceptive Burden)\n');
fprintf('Bootstrapped OR: %.3f\n', bootOR_PC1);
fprintf('Empirical 95%% CI: [%.3f - %.3f]\n', ci_boot_pc1(1), ci_boot_pc1(2));
fprintf('Empirical p-value: %.4f\n', p_boot_pc1);
fprintf('----------------------------------------\n');
%% 7. TABLE REPRODUCTION (TABLE 1 & TABLE 2)
fprintf('\n--- DATA FOR TABLE 1: DEMOGRAPHICS & CLINICAL CHARACTERISTICS ---\n');
fprintf('Sample Size: n = %d\n', height(data));
fprintf('Age (Mean ± SD): %.1f ± %.1f\n', mean(data.Age), std(data.Age));
fprintf('Female Sex: n = %d (%.1f%%)\n', sum(pred_sex), (sum(pred_sex)/height(data))*100);
fprintf('Muscle Pain: n = %d (%.1f%%)\n', sum(m_pain), (sum(m_pain)/height(data))*100);
fprintf('Joint Pain: n = %d (%.1f%%)\n', sum(j_pain), (sum(j_pain)/height(data))*100);
fprintf('Intra-articular Disorder: n = %d (%.1f%%)\n', sum(i_disorder), (sum(i_disorder)/height(data))*100);
fprintf('Neck Pain Presence: n = %d (%.1f%%)\n', sum(target_y), (sum(target_y)/height(data))*100);

fprintf('\n--- DATA FOR TABLE 2: BIVARIATE ASSOCIATIONS (OR) ---\n');
% Helper function to calculate simple bivariate OR
calcOR = @(var) exp(fitglm(var, target_y, 'Distribution', 'binomial').Coefficients.Estimate(2));

fprintf('Bivariate OR for Muscle Pain: %.2f\n', calcOR(m_pain));
fprintf('Bivariate OR for Joint Pain: %.2f\n', calcOR(j_pain));
fprintf('Bivariate OR for Intra-articular: %.2f\n', calcOR(i_disorder));
