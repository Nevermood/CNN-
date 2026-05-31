% Шаг 1. Загрузить обученную модель (если она ещё не в рабочей области)
load('Custom_CNN.mat', 'Custom_CNN');

% Шаг 2. Создать imageDatastore для валидационного набора
valDatastore = imageDatastore('Dataset/val', ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Определяем все уникальные метки (классы)
allLabels = valDatastore.Labels;
uniqueLabels = unique(allLabels);

% Шаг 3. Для каждого класса выберем по 2 изображения
for c = 1:numel(uniqueLabels)
    
    % Текущий класс
    currentLabel = uniqueLabels(c);
    
    % Индексы всех изображений, принадлежащих этому классу
    idx = find(allLabels == currentLabel);
    
    % Сколько изображений хотим взять (максимум 2, но может быть меньше, если валидационных изображений меньше)
    numToTest = min(2, numel(idx));
    
    % Берём первые (numToTest) изображений
    for i = 1:numToTest
        % Считываем i-е изображение из списка
        I = readimage(valDatastore, idx(i));
        
        % Изменяем размер под сеть (например, 128x128)
        I_resized = imresize(I, [128 128]);
        
        % Предсказываем метку
        predictedLabel = classify(Custom_CNN, I_resized);
        
        % Отображаем изображение и выводим предсказанную и истинную метку
        figure;
        imshow(I);
        title(sprintf("Класс: %s\nПредсказано: %s", ...
            string(currentLabel), string(predictedLabel)));
    end
end
