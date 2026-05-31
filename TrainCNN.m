% Задание путей к папкам с данными
trainFolder = fullfile('Dataset', 'train');
valFolder = fullfile('Dataset', 'val');

% Создание datastore для тренировочных и валидационных данных
Training_Data = imageDatastore(trainFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
Validation_Data = imageDatastore(valFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Определение входного размера изображений (например, 128x128 с 3 цветными каналами)
inputSize = [128 128 3];

% Определение количества классов на основе меток
numClasses = numel(categories(Training_Data.Labels));

% Создание augmentedImageDatastore для автоматического изменения размера изображений
Resized_Training_Data = augmentedImageDatastore(inputSize(1:2), Training_Data);
Resized_Validation_Data = augmentedImageDatastore(inputSize(1:2), Validation_Data);

% Определение архитектуры сети
layers = [
    imageInputLayer(inputSize, 'Name', 'input')
    
    convolution2dLayer(3, 8, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(3, 'Stride', 2, 'Name', 'maxpool1')
    
    convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(3, 'Stride', 2, 'Name', 'maxpool2')
    
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')
    maxPooling2dLayer(3, 'Stride', 2, 'Name', 'maxpool3')
    
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv4')
    batchNormalizationLayer('Name', 'bn4')
    reluLayer('Name', 'relu4')
    maxPooling2dLayer(3, 'Stride', 2, 'Name', 'maxpool4')
    
    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv5')
    batchNormalizationLayer('Name', 'bn5')
    reluLayer('Name', 'relu5')
    maxPooling2dLayer(3, 'Stride', 2, 'Name', 'maxpool5')
    
    
    fullyConnectedLayer(512, 'Name', 'fc1')
    reluLayer('Name', 'relu6')
    dropoutLayer(0.5, 'Name', 'dropout')
    fullyConnectedLayer(numClasses, 'Name', 'fc2')
    
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

% Определение параметров обучения
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 4, ...              % размер мини-пакета
    'MaxEpochs', 10, ...                 % количество эпох
    'InitialLearnRate', 4e-4, ...          % начальная скорость обучения
    'Shuffle', 'every-epoch', ...
    'ValidationData', Resized_Validation_Data, ...
    'ValidationFrequency', 3, ...         % частота валидации
    'Verbose', false, ...
    'Plots', 'training-progress');

% Обучение сети
Custom_CNN = trainNetwork(Resized_Training_Data, layers, options);

% Сохранение обученной модели
save('Custom_CNN.mat', 'Custom_CNN');
