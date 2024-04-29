%clear the workspace
clear all;
close all;
clc;

%set folder path to the subjects datasets
myFolder = 'C:\Users\code development\Downloads\combined\combined_data';
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

filePattern = fullfile(myFolder, '*.mat');
theFiles = dir(filePattern);
all_acc = [];
sum = 0;
numFiles = length(theFiles);
for i = 1:numFiles
    baseFileName = theFiles(i).name;
    fullFileName = fullfile(theFiles(i).folder, baseFileName);
    fprintf(1, 'Now reading subjects files %s\n', fullFileName);
    load(fullFileName); %load each mat file into "data"
    %data = data(9:32,:,:);
    %data = fnormal(data);
    name = strcat('subject',num2str(i));
    variable.(name) = data;
    sum = sum+data;
end

norm_sum = sum/numFiles;

%CROSS VALIDATION LOOCV
for p = 1:numFiles
    x = [];
    y = [];
    %load and read the data to 3 classes seperately from 27 subject for training
    class_1 = [];
    class_2 = [];
    class_3 = [];
    
    %initialize empty arrays to contain the cov matrices of each trails
    covariance1 = [];
    covariance2 = [];
    covariance3 = [];
    covariance = [];
    disp('-------------------');
    disp(['Starting cross validation for subject', num2str(p)]);
    
    for j = 1:length(theFiles)
        if (j ~=p)
            
            baseFileName = theFiles(j).name;
            fullFileName = fullfile(theFiles(j).folder, baseFileName);
            fprintf(1, 'Now reading %s\n', fullFileName);
            load(fullFileName); %load each mat file into "data"
            %label = squeeze(data(33,1,1:32));
            %save('label.mat', 'label');
            %label = diag(label);
            %save('label.mat', 'label');
            %find the first label ,13Hz , trail locations
            first_f = ismember(label, 13); % return an array with zeros and ones indicating the label
            first = find(diag(first_f * label),13); %returns an array with the first label trials locations
            %find the second label , 17 Hz, trail locations
            second_f = ismember(label, 17); % return an array with zeros and ones indicating the label
            second = find(diag(second_f * label),17); %returns an array with the second label trials locations
            % find the third label trail locations , 21 Hz
            third_f = ismember(label, 21); % return an array with zeros and ones indicating the label
            third = find(diag(third_f * label),21); %returns an array with the third label trials locations
            %Extract the classes
            class1 = data(first,:,:); % extract the 13Hz trails ina single matrix
            class2 = data(second,:,:); %extract the 17Hz trails ina single matrix
            class3 = data(third,:,:); %extract the 21Hz trails ina single matrix
            class_1 = [class_1;class1];
            class_2 = [class_2; class2];
            class_3 = [class_3; class3];
        end
    end
    disp(length(theFiles))
    % FOR FIRST FREQUENCY / CLASS_1 13 HZ
    for i = 1:251
        data1 = transpose(class_1(i,:));
        data1 = BPF([12 14], [16 18], [20 22], data1); % apply band pass filter
        %data = data/norm_sum; %normalize the data over each trail
        cova = covar(data); %find the covariance matrix
        covariance1 = cat(3, covariance1, cova);
        %save('class1_cov.mat','covariance1');
    end
    
    %FOR FIRST FREQUENCY / CLASS_2 17HZ
    for i = 1:251
        data1 = transpose(class_2(i,:));
        data1 = BPF([12 14], [16 18], [20 22], data1); % apply band pass filter
%        data = data/norm_sum; %normalize the data over each trail
        cova = covar(data1); %find the covariance matrix
        covariance2 = cat(3, covariance2, cova);
        %save('class2_cov.mat','covariance2');
    end
    
    %FOR THIRD FREQUENCY / CLASS_3 21HZ
    for i = 1:251
        disp(length(data))
        data1 = transpose(class_3(i,:));
        data1 = BPF([12 14], [16 18], [20 22], data1); % apply band pass filter
     %   data = data/norm_sum; %normalize the data over each trail
        cova = covar(data1); %find the covariance matrix
        covariance3 = cat(3,covariance3, cova);
        %save('class3_cov.mat','covariance3');
    end
    
    %NOW find the three mean covariance matrices for each class
    class_1_mean = riemann_mean(covariance1); %class1 mean
    class_2_mean = riemann_mean(covariance2); %class2 mean
    class_3_mean = riemann_mean(covariance3); %class3 mean
    
    %TEST THE testing sub with the mean of covriance ma
    % trices of the three
    %classes using MDM
    
    %%%%%%%% LOADING THE TESTING DATA FILE %%%%%%%%%%%%%%%%%
    baseFileName = theFiles(p).name;
    fullFileName = fullfile(theFiles(p).folder, baseFileName);
    fprintf(1, 'Loading the testing subject data %s\n', fullFileName);
    load(fullFileName); %load each mat file into "data"
    %true_l = squeeze(data(33,1,1:32));
    %true_l = true_l(9:32,:);
    true_l = label;
    % loop over the testing subject data to find the covariance for each
    % trail of the 24 trails eleminating the first 8 trails of resting

    dat = transpose(data);
    dat = BPF([12 14], [16 18], [20 22], dat); % apply band pass filter
    %dat = dat/norm_sum; %normalize the data over each trail
    cova = covar(dat); %find the covariance matrix
    covariance = cat(3,covariance, cova); %this will be 24x24x24? cov for all test subject sessions of 24 trail
    %save('class3_cov.mat','covariance3');
    
    %calculate the dist ( prediction of label) of each of the subject
    %sessions of 24 and get the accuracy after checking with the true label
    %The Accuracy of the model is the average of the accuracy of each fold.
    predictions = [];
    %true_label = []; %create an empty array for true
    for i = 1:251
        x = squeeze(covariance(:,:,i));
        t = predict(x, class_1_mean, class_2_mean, class_3_mean);
        predictions = [predictions; t];
        % true_label = test_label(i,:);
        %true_label = [true_label; true_label];
        
    end
        %count the correct label
        count = 0;
        for k = 1:251
            if double(predictions(k)) == double(true_l(k))
                count = count +1;
            end
        end
    
        acc = count/251;
    
    %acc = accuracy_score(true_l, predictions);
    all_acc = [all_acc; acc];
    fprintf(1, 'Accuracy: %s\n', acc);
end

avg_acc= mean(all_acc);
fprintf(1, 'Average Accuracy is : %s\n', avg_acc);

