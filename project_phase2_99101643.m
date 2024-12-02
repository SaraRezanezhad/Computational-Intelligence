evolutionaryAlgorithm(NormalizingFeatures,TrainLabels,Layer,lable1,lable2,100,20,300);
function evolutionaryAlgorithm(NormalizingFeatures,TrainLabels,Layer,ind1,ind2,pop_num,numOfFeatures,numOfSamples)
    
    % جمعیت اولیه و محاسبه‌ی سازگاری (fitness) اعضا
    InitialPop = makeInitPopulation(pop_num, numOfFeatures, numOfSamples);
    InitialFit = calculateFitness(InitialPop, NormalizingFeatures, TrainLabels, Layer, ind1, ind2);

    % انتخاب و الگوهای هجومی
    [pop, fit] = selectPatterns(InitialPop, InitialFit);

    % اجرای الگوریتم تکاملی
    for iteration = 1:10
        [pop, fit] = applyEvolutionaryOperators(pop, fit, NormalizingFeatures, TrainLabels, Layer, ind1, ind2);
    end

    % بهترین جمعیت و تولید مدل مورد نظر
    bestIndividual = pop(19,:);
    InputData = NormalizingFeatures(bestIndividual, 1:550);
    net = feedforwardnet(Layer);
    net = train(net, InputData, TrainLabels);
    view(net)
    mlpLabels = 2 * heaviside(net(NormalizingFeatures(bestIndividual, 551:end))) - 1;
    save('MLP2.mat','mlpLabels');
    
    TrainingData = NormalizingFeatures(I,1:550);
    net = newrb(TrainingData, TrainLabels, 0, 1, 59, 25);
    y = sim(net, TrainingData);
    labels = 2 * (y >= 0) - 1;
    RBF_BestPerformance = 1 - sum(abs(labels - TrainLabels)) / 2 / 550;
    RBF_Labels2 = 2 * (sim(net, NormalizingFeatures(I, 551:709)) >= 0) - 1;
    save('RBF2.mat', 'RBF_Labels2');


end

function initPopulation = makeInitPopulation(populationSize, numOfFeatures, numOfSamples)
    initPopulation = randi([1, numOfSamples], populationSize, numOfFeatures);
end

function fitnessValues = calculateFitness(population, Y, TrainLabels, Layer, ind1, ind2)
    fitnessValues = zeros(size(population, 1), 1);
    for i = 1:size(population, 1)
        fitnessValues(i) = fitness(Y, population(i, :), TrainLabels, Layer, ind1, ind2);
    end
end

function [selectedPopulation, selectedFitness] = selectPatterns(population, fitnessValues)
    [~, sortedIndices] = sort(fitnessValues);
    selectedIndices = sortedIndices(1:10);
    selectedPopulation = population(selectedIndices, :);
    selectedFitness = fitnessValues(selectedIndices);
end

function [newPopulation, newFitness] = applyEvolutionaryOperators(population, fitnessValues, Y, TrainLabels, Layer, ind1, ind2)
    newPopulation = population;
    newFitness = fitnessValues;
    for i = 1:2:size(population, 1)
        [new1, new2] = crossover(population(i, :), population(i+1, :));
        newPopulation = [newPopulation; new1; new2];
        newFitness = [newFitness; fitness(Y, new1, TrainLabels, Layer, ind1, ind2); fitness(Y, new2, TrainLabels, Layer, ind1, ind2)];
    end
end

function [child1, child2] = crossover(parent1, parent2)
    N = floor(length(parent1) / 2);
    child1 = [parent1(1:N), parent2(N+1:end)];
    child2 = [parent2(1:N), parent1(N+1:end)];
end

function fit = fitness(NormalizingFeatures, I, TrainLabels, Layer, ind1, ind2)
    perf = perMlp(NormalizingFeatures,I,TrainLabels, Layer);
    Fisher_ratio = J(NormalizingFeatures, I, ind1, ind2);
    fit = perf + 50 * Fisher_ratio;
end

function perf = perMlp(NormalizingFeatures,I,TrainLabels, Layer)
  accuracy = [];
foldSize = floor(550 / 5);
% K-fold cross validation for MLP
TrainingData = NormalizingFeatures(I,1:550);
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
perf = mean(accuracy);

end

function Fisher_ratio = J(NormalizingFeatures, I, ind1, ind2)
    f = NormalizingFeatures(I,1:550);
    mu1 = mean(f(:,ind1), 2);
    mu2 = mean(f(:,ind2), 2);
    mu0 = mean(f, 2);
    
    c1 = 0;
    for i = 1:length(ind1)
        c1 = c1 + (f(:,ind1(i)) - mu1) * (f(:,ind1(i)) - mu1).';
    end
    
    c2 = 0;
    for i = 1:length(ind2)
        c2 = c2 + (f(:,ind2(i)) - mu2) * (f(:,ind2(i)) - mu2).';
    end
    
    N1 = length(ind1);
    N2 = length(ind2);
   
    
   Sb = (mu1 - mu0) * (mu1 - mu0).' + (mu2 - mu0) * (mu2 - mu0).';
    Sw = c1/N1 + c2/N2;
    Fisher_ratio = trace(Sb) / trace(Sw);
end

