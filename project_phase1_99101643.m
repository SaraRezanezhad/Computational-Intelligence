%% الف
fs = 1000;
data = cat(3,TrainData,TestData);

% Statistical Features
Var = reshape(var(data,0,2),59,709); %1-59
Mean =reshape(mean(data,2),59,709);  %60-118

rms_value = reshape(rms(data,2),59,709);
FormFactor = rms_value ./ Mean; %119-177

features = [Mean;Var;FormFactor];  

% Frequency Features
for j=1:59                          
        Meanf = meanfreq(reshape(data(j,:,:),709,5000)',fs);
        features = [features;Meanf]; %178-236
end

for j=1:59                          
        Medf = medfreq(reshape(data(j,:,:),709,5000)',fs);
        features = [features;Medf]; %237-295
end

FrequencyRange = [0 4 8 12 16 21 100];
for i=1:6                           
    range = [FrequencyRange(i) FrequencyRange(i+1)];
    for j=1:59
        BandPower = bandpower(reshape(data(j,:,:),709,5000)',fs,range); %296-649 (59*6)
        features = [features;BandPower];
    end
end

NormalizingFeatures = normalize(features);       % normalizing feature matrix

%% ب
lable1 = find(TrainLabels>0);
lable2 = find(TrainLabels<0);

% Calculating J and top 10 features in Statistical Features
J1 = zeros(177, 1);
for i = 1:177
    m0 = mean(NormalizingFeatures(i,:));
    m1 = mean(NormalizingFeatures(i,lable1));
    m2 = mean(NormalizingFeatures(i,lable2));
    v1 = var(NormalizingFeatures(i,lable1));
    v2 = var(NormalizingFeatures(i,lable2));
    fisher = abs((sqrt(m1-m0)+sqrt(m2-m0))/(v1+v2));
    J1(i) = fisher;
end
[~, I1] = maxk(J1, 10);
BestFeatures1 = J1(I1); %Mean!

% Calculating J and top 10 features in Frequency Features
J2 = zeros(472, 1);
for i = 178:649
    m0 = mean(NormalizingFeatures(i,:));
    m1 = mean(NormalizingFeatures(i,lable1));
    m2 = mean(NormalizingFeatures(i,lable2));
    v1 = var(NormalizingFeatures(i,lable1));
    v2 = var(NormalizingFeatures(i,lable2));
    fisher = abs((sqrt(m1-m0)+sqrt(m2-m0))/(v1+v2));
    J2(i-177) = fisher;
end
[~, I2] = maxk(J2, 10);
BestFeatures2 = J2(I2); %BandPower!
I = [I1;I2];
%% ج
TrainingData = NormalizingFeatures(I,1:550);
perf = [];    
t =TrainLabels;
for i=1:15
    net = feedforwardnet(i);
    net = train(net,TrainingData,TrainLabels);
    y = net(TrainingData);
    %perf = perform(net,t,y);
    perf = [perf;perform(net,t,y)];
end
[pp, Layer] = max(perf);

%% ادامه ج
net = feedforwardnet(Layer);
net = train(net,TrainingData,TrainLabels);
MLPLabels = 2*heaviside(net(NormalizingFeatures(I,551:709)))-1;
save('MLP1.mat','MLPLabels');
%% ادامه ج!
net = feedforwardnet(Layer);
net = train(net,TrainingData,TrainLabels);
mlplabels = 2*heaviside(net(NormalizingFeatures(I,551:709)))-1;
accuracy = [];
foldSize = floor(550 / 5);
% K-fold cross validation for MLP
cv = cvpartition(size(TrainingData, 1), 'KFold', 5);
validationIdx = 1:5:550;
for i=1:cv.NumTestSets
    inpp = TrainingData;
    inpp(:,validationIdx)=[];
    vall = TrainingData(:,validationIdx);
    TrainLabel1 = TrainLabels;
    TrainLabel1(validationIdx)=[];
    net = feedforwardnet(Layer);
    net = train(net,inpp,TrainLabel1);
    mlplabels = 2*heaviside(net(vall))-1;
    accuracy = [accuracy 1-sum(abs(mlplabels-TrainLabels(validationIdx)))/2/foldSize];
    validationIdx = validationIdx + ones(1,foldSize);
end
average_accuracy = mean(accuracy);

%% د
net = newrb(TrainingData, TrainLabels, 0, 1, 59, 25);
y = sim(net, TrainingData);
labels = 2 * (y >= 0) - 1;
RBF_BestPerformance = 1 - sum(abs(labels - TrainLabels)) / 2 / 550;

RBF_Labels = 2 * (sim(net, NormalizingFeatures(I, 551:709)) >= 0) - 1;
save('RBF1.mat', 'RBF_Labels');

