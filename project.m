% code to run method three in parallel

clear all;
close all;
clc;
tic
accuracy = []
subject = [1:1:28];
n = length(subject);
threshold_scheduler = ThresholdScheduler(99, 0.1, 97, 99);
zulabel = [13, 17, 21];
thr = 99; %make it random or based on NN?
myFolder = "C:\Users\code development\OneDrive - University of Central Florida\TAX2024\DataAnalysis_Manifolds\dataset";
%check if the folder exist if not display warning
if ~isfolder(myFolder)
    errorMessage = sprintf('Error: The following folder does not exist:\n%s\nPlease specify a new folder.', myFolder);
    uiwait(warndlg(errorMessage));
    myFolder = uigetdir(); %ask for a new one
    if myFolder == 0
        %user click cancel
        return;
    end
end
summ = 0;
filePattern = fullfile(myFolder, '*.mat');
theFiles = dir(filePattern);
for i = 1:28
    baseFileName = theFiles(i).name;
    fullFileName = fullfile(theFiles(i).folder, baseFileName);
    fprintf(1, 'Now reading subjects files %s\n', fullFileName);
    load(fullFileName); %load each mat file into "data"
    %data = rescale(data);
    data = fnormal(data);
    name = strcat('subject',num2str(i));
    variable.(name) = data;
    summ= summ+data;
end
norm_sum = summ/28;
epoch = 0;

figure;
for s = 1:n
    epoch = epoch +1;
    baseFileName = theFiles(s).name;
    fullFileName = fullfile(theFiles(s).folder, baseFileName);
    fprintf(1, 'Now reading subjects files %s\n', fullFileName);
    s
    load(fullFileName);

    %%%%

    labels = squeeze(data(33,1,1:32));
    %data = rescale(data);
    %data = data(1:32,:,:);
    data = data(1:32,:,513:1280);
    normal = fnormal(data); %get the normalization factor of the current subject
    data = data / norm_sum;
    %hyper parameter selection

    filtered_bands = [];
    cat_filtered_bands = [];
    labels_new =[];
    for k = 1:32
        if labels(k) ~= 0 %remove rest trials
            labels_new = [labels_new, labels(k)];
            sig = squeeze(data(k,:,:));
            sig = transpose(sig);
            bands = BP([12.95 13.05], [16.9 17.1], [20.9 21.1], sig);
            filtered_bands = cat(3,filtered_bands, bands); %gives (8*3) = 24, (768,24)
        end
    end
    %ends up (samples, features 24, trials)
    % concatinate the bands data to prepare for PCA! row= observations, col=var
    for i = 1:24
        datat = squeeze(filtered_bands(:,:,i));
        cat_filtered_bands = cat(1,cat_filtered_bands, datat);
    end
    %end up being (++samples, 24 features) (24576x24)
    % now that we have a contactenated data all we need is to find the PCA
    % of each band and apply it to the dataset
    % should i cat the bands? before finding PCA tryy = cat(2, cat_theta, cat_alpha, cat_beta, cat_gamma);
    % or do PCA seperately and then cat them? to proceed.

    PCA_DATA = [];

    [coeff,score,latent,tsquared,explained,mu] = pca(cat_filtered_bands);
    idx = find(cumsum(explained)>thr,1);
    if idx == 1
        idx = idx+1;
    end
    for i = 1:24
        th = squeeze(filtered_bands(:,:,i));
        c = th*coeff(:,1:idx); %apply the weights to the filtered data
        PCA_DATA = cat(3, PCA_DATA, c); %reconstruct the data
    end
    %PCAData (samples, features, trials)

    % next step is to concatinate all the channels we got 34 channel in total

    %initialize the empty matrix we want for each sub
    predictions = [];
    true_labels = [];
    for t = 1:24
        cov_c1 = [];
        cov_c2 = [];
        cov_c3 = [];
        %cov_rest = [];
        cov_train = [];
        for v = 1:24
            if (v~=t)
                sig = squeeze(PCA_DATA(:,:,v));
                [sigma,rho] = shrinkage_cov(sig);
                cov_train = cat(3, cov_train,sigma); %24x24

                labels_t = labels_new(v);
                if labels_t == 17
                    cov_c2 = cat(3, cov_c2, sigma);
                elseif labels_t == 13
                    cov_c1 = cat(3, cov_c1, sigma);
                elseif labels_t == 21
                    cov_c3 = cat(3, cov_c3, sigma);
                    % else
                    %     cov_rest = cat(3, cov_rest, sigma);
                end
            end
        end
        %calculate the mean of each class!
        class_1_mean = riemann_mean(cov_c1);
        class_2_mean = riemann_mean(cov_c2);
        class_3_mean = riemann_mean(cov_c3);
        %class_rest_mean = riemann_mean(cov_rest);
        % START CROSS VAIDATION TESTING!
        sigt = squeeze(PCA_DATA(:,:,t));
        [cov_test,rho] = shrinkage_cov(sigt);
        % seperate if neg or pos
        t_labelss = labels_new(1,t); %get the true labels
        true_labels = [true_labels; t_labelss]; %append to the list for acc
        pred = predict(cov_test, class_1_mean, class_2_mean, class_3_mean);
        predictions = [predictions; pred];
    end

    % Generate confusion matrix with labels
    C = confusionmat(true_labels, predictions);

    % Plot confusion matrix as a subplot
    subplot(5, 5, epoch);
    imagesc(C);
    %title(['session ', num2str(epc)]);
    colorbar;

    % Add text to each square
    for i = 1:size(C, 1)
        for j = 1:size(C, 2)
            text(j, i, num2str(C(i, j)), 'HorizontalAlignment', 'center', FontSize=5);
        end
    end

    % Set the x and y tick labels to the class labels
    xticks(1:length(zulabel));
    yticks(1:length(zulabel));
    xticklabels(zulabel);
    yticklabels(zulabel);
    acc = accuracy_score(true_labels, predictions);
    accuracy = [accuracy, acc];
    acc
    threshold_scheduler = threshold_scheduler.on_epoch_end(epoch, 23);
end

% parfor s = 1:n
%     data = load_data(s, theFiles);
%     [acc] = project_func(data, th);
%     accuracy(s,1)= acc;
% end
% ac = mean(accuracy);
% fprintf(1, 'Average Accuracy is : %s\n', ac);

end
ac = mean(accuracy);
fprintf(1, 'Average Accuracy is : %s\n', ac);
toc
function [data] =load_data(s, theFiles)
baseFileName = theFiles(s).name;
fullFileName = fullfile(theFiles(s).folder, baseFileName);
fprintf(1, 'Now reading subjects files %s\n', fullFileName);
[data]= load(fullFileName);
end

