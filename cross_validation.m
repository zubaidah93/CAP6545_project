%zubaidah almashhadani
%PID 5411909

clear all;
close all;
clc;

fs = 256;
all_acc = [];
%set folder path to the subjects datasets
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
%Get list of all files in the folder with the desired name pattern
true = [];
false =[];
zulabel = [13, 17, 21];


sum = 0;
filePattern = fullfile(myFolder, '*.mat');
theFiles = dir(filePattern);

for i = 1:28
    baseFileName = theFiles(i).name;
    fullFileName = fullfile(theFiles(i).folder, baseFileName);
    fprintf(1, 'Now reading subjects files %s\n', fullFileName);
    load(fullFileName); %load each mat file into "data"
    
    data = fnormal(data);
    name = strcat('subject',num2str(i));
    variable.(name) = data;
    sum= sum+data;
end
norm_sum = sum/28;
figure;
%loop over all the subjects
for s = 1:28
    
    disp('-------------------');
    disp(['Starting cross validation for the test subject', num2str(s)]);
    %initialize empty matrix
    class_1_cov = [];
    class_2_cov = [];
    class_3_cov = [];
    covariance = [];
    predictions = [];
    count = 0;
    for j = 1:length(theFiles)
        
        if (j ~=s)
            cov_mat = []; %contain the covariance matrices for all the trials = 32, (24x24)
            baseFileName = theFiles(j).name;
            fullFileName = fullfile(theFiles(j).folder, baseFileName);
            fprintf(1, 'Now reading %s\n', fullFileName);
            load(fullFileName); %load each mat file into "data"
            LABELS = squeeze(data(33,1,1:32));
            %save('LABELS.mat', 'LABELS');
            labels = diag(LABELS);
            data = data(1:32,:,513:1280); %take only 3 seconds , wait first 2 seconds and remove the labels part trial
            %here the data is 32x24x768 which is correct
            normal = fnormal(data); %get the normalization factor of the current subject
            data = data / normal; %normalize the data of the trials for the current subject
            %get the covariance matrix for the whole subject of all the 32 trials
            for i = 1:32
                siga = squeeze(data(i,:,:));
                sigb = transpose(siga);
                sig = BPF([12.95 13.05], [16.9 17.1], [20.9 21.1],sigb);
                [sigma,rho] = shcovft(sig);
                cov_mat = cat(3,cov_mat,sigma);
            end
            first_f = ismember(labels, 13); % return an array with zeros and ones indicating the label
            first = find(diag(first_f * labels),13); %returns an array with the first label trials locations
            %find the second label , 17 Hz, trail locations
            second_f = ismember(labels, 17); % return an array with zeros and ones indicating the label
            second = find(diag(second_f * labels),17); %returns an array with the second label trials locations
            % find the third label trail locations , 21 Hz
            third_f = ismember(labels, 21); % return an array with zeros and ones indicating the label
            third = find(diag(third_f * labels),21); %returns an array with the third label trials locations
            %since cov_mat is 24x24x32 matrix we extract the classes as:
            class1_cov = cov_mat(:,:,first); %resulting in 24x24x8
            class2_cov = cov_mat(:, :, second);
            class3_cov = cov_mat(:,:, third);
            
            %now concatinate the matrices together to get 24x24x216
            class_1_cov = cat(3, class_1_cov, class1_cov);
            class_2_cov = cat(3, class_2_cov, class2_cov);
            class_3_cov = cat(3, class_3_cov, class3_cov);
            
        end
        
    end
    
    %note that riemann_mean() takes in NxNxK matrix where the matrix is NxN
    class_1_mean = riemann_mean(class_1_cov);
    class_2_mean = riemann_mean(class_2_cov);
    class_3_mean = riemann_mean(class_3_cov); % a 24x24 matrix
    % NOW draw the means for all the subject
    % figure;
    % subplot(2,2,1);image(class_1_mean/max(max(class_1_mean))*255);title("13 Hz");
    % subplot(2,2,2);image(class_2_mean/max(max(class_2_mean))*255); title("17 Hz");
    % subplot(2,2,3);image(class_3_mean/max(max(class_3_mean))*255);title("21 Hz");
    % sgtitle("The means for all subjects except : " + s) ;
    % 
    
    %%%%%%%% LOADING THE TESTING DATA FILE %%%%%%%%%%%%%%%%%
    
    baseFileName = theFiles(s).name;
    fullFileName = fullfile(theFiles(s).folder, baseFileName);
    fprintf(1, 'Loading the testing subject data %s\n', fullFileName);
    load(fullFileName); %load each mat file into "data"
    
    
    lab = squeeze(data(33,1,1:32));
    test_labels = diag(lab);
    data = data / norm_sum;
    
    % loop over the testing subject data to find the covariance for each
    % trail of the 24 trails eleminating the first 8 trails of resting
    for m = 1:32
        dat = transpose(squeeze(data(m,:,:)));
        %dat = BPF([12.95 13.05], [16.9 17.1], [20.9 21.1], dat); % apply band pass filter
        dat = BPF([12 14], [16 18], [20 22], dat);
        [cova,rho] = shcovft(dat); %find the covariance matrix
        covariance = cat(3,covariance, cova); %this will be 24x24x24? cov for all test subject sessions of 24 trail
    end
    test_classes = [];
    train_cov = [];
    first_f = ismember(test_labels, 13); % return an array with zeros and ones indicating the label
    first = find(diag(first_f * test_labels),13); %returns an array with the first label trials locations
    %find the second label , 17 Hz, trail locations
    second_f = ismember(test_labels, 17); % return an array with zeros and ones indicating the label
    second = find(diag(second_f * test_labels),17); %returns an array with the second label trials locations
    % find the third label trail locations , 21 Hz
    third_f = ismember(test_labels, 21); % return an array with zeros and ones indicating the label
    third = find(diag(third_f * test_labels),21); %returns an array with the third label trials locations
    first_cov = covariance(:,:,first); %resulting in 24x24x8
    second_cov = covariance(:, :, second);
    third_cov = covariance(:,:, third);
    
    train_cov = cat(3,train_cov, first_cov);
    train_cov = cat(3,train_cov, second_cov);
    train_cov = cat(3,train_cov, third_cov);
    
    first_c = [13;13;13;13;13;13;13;13];
    second_c= [17;17;17;17;17;17;17;17];
    third_c = [21;21;21;21;21;21;21;21];
    
    test_classes = [test_classes ; first_c];
    test_classes = [test_classes ; second_c];
    test_classes = [test_classes ; third_c];
    
    %now start prediction step
    for i = 1:24
        x = squeeze(train_cov(:,:,i));
        t = predict(x, class_1_mean, class_2_mean, class_3_mean);
        predictions = [predictions; t];
        if t == (test_classes(i))
            count = count +1;
            true = [true; t];
        else
            false = [ false; test_classes(i)];
        end
    end
    true_labels = test_classes;
    predicted_labels = predictions;
     % Generate confusion matrix with labels
    C = confusionmat(true_labels, predicted_labels);

    % Plot confusion matrix as a subplot
    subplot(5, 6, s);  
    imagesc(C);
    title(['Subject ', num2str(s)]);
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

    acc = count/24;
    all_acc = [all_acc; acc];
    fprintf(1, 'Accuracy: %s\n', acc);
    s
    acc
end

avg_acc= mean(all_acc);
fprintf(1, 'Average Accuracy is : %s\n', avg_acc);

