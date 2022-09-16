% Assignment 1
clc
close all
clear all

% Global variables
hidden_nodes = 2;
max_epochs = 200;
eta = 0.001;
seed = 5;
alpha = 0.9;

% Seeting the seed
rng(seed)

% Generate dataset
ndata = 100;
mA = [ 1.0, 0.5]; sigmaA = 0.2;
mB = [ 0.0, -0.1]; sigmaB = 0.2;
classA(1,:) = [ randn(1,round(0.5*ndata)) .* sigmaA - mA(1), ...
randn(1,round(0.5*ndata)) .* sigmaA + mA(1)];
classA(2,:) = randn(1,ndata) .* sigmaA + mA(2);
classB(1,:) = randn(1,ndata) .* sigmaB + mB(1);
classB(2,:) = randn(1,ndata) .* sigmaB + mB(2);

% Plot dataset
figure()
scatter(classA(1,:), classA(2,:), [], "red") 
hold on
scatter(classB(1,:), classB(2,:), [], "blue")

% Define the patterns and labels
classA_target = ones(1, ndata);
classB_target = -ones(1, ndata);

% Add the targets
classA = [classA ; classA_target];
classB = [classB ; classB_target];

% 3.1.1 Classification of non-lineary seperable data

%% Subsection 1: Adjusting hidden nodes and comparing MSE
data = [classA, classB];

[MSE_vec, misclass_vec] = MLPbackprop(data, hidden_nodes, max_epochs,eta,true,seed,alpha);

errorplot(1:max_epochs, MSE_vec, misclass_vec, 'Errors vs epochs', 'Number of epochs');

nodes_vec = 2:10;
MSE_nodes = [];
misclass_nodes = [];

for node = nodes_vec
 [MSE_vec, misclass_vec] = MLPbackprop(data, node, max_epochs,eta,true,seed,alpha);
 MSE_nodes = [MSE_nodes, MSE_vec(end)];
 misclass_nodes = [misclass_nodes, misclass_vec(end)];

end

errorplot(nodes_vec, MSE_nodes, misclass_nodes, 'Errors vs number of hidden nodes', 'Number of hidden nodes');

%% Subsection 2.1: Subsampling 25% of each class

% 25% training data from each class
% Generation of random indexes
indexA25 = datasample(1:100, 25, 'Replace', false);
indexB25 = datasample(1:100, 25, 'Replace', false);

% Validation set
classA25 = classA(:,indexA25);
classB25 = classB(:,indexB25);

% Training set
classA75 = classA;
classA75(:,indexA25) = [];
classB75 = classB;
classB75(:,indexB25) = [];


% Training the neural network
data = [classA75, classB75];
[MSE_vec, misclass_vec, W, V] = MLPbackprop(data, hidden_nodes, max_epochs,eta,true,seed,alpha);

% Select validation set and compute errors
n = 50;
data25 = [classA25, classB25];
[MSE, misclass] = validation(W, V, data25);

valid_MSE = MSE;
train_MSE = MSE_vec(end);

valid_misclass = misclass;
train_misclass = misclass_vec(end);

% Plotting
errorplot(1:max_epochs, MSE_vec,misclass_vec,'25% from each class','Number of epochs')

hold on
xx = [max_epochs, max_epochs];
yy = [valid_MSE, valid_misclass];
h2 = plot(xx(2), yy(2), '*', 'LineWidth', 3);
set(h2, 'Color', 'blue');

hold on;
yyaxis right
h1 = plot(xx(1), yy(1), '*', 'LineWidth', 3);
set(h1, 'Color', 'red');

legend({'Ratio of missclassifications','Validation misclassifications', 'MSE', 'Validation MSE'});









%% Subsection 2.2: Subsampling 50% from  class A

% 50% training data from class A
% Generation of random indexes
indexA50 = datasample(1:100, 50, 'Replace', false);

% Validation set
classA50 = classA(:,indexA50);

% Training set
classAtrain = classA;
classAtrain(:,indexA50) = [];
classBtrain = classB;


% Training the neural network
data = [classAtrain, classBtrain];
[MSE_vec, misclass_vec, W, V] = MLPbackprop(data, hidden_nodes, max_epochs,eta,true,seed,alpha);

% Select validation set and compute errors
n = 50;
data50 = classA50;
[MSE, misclass] = validation(W, V, data50);

valid_MSE = MSE;
train_MSE = MSE_vec(end);

valid_misclass = misclass;
train_misclass = misclass_vec(end);

% Plotting
errorplot(1:max_epochs, MSE_vec,misclass_vec,'50% from class A','Number of epochs')
hold on

xx = [max_epochs, max_epochs];
yy = [valid_MSE, valid_misclass];
h2 = plot(xx(2), yy(2), '*', 'LineWidth', 3);
set(h2, 'Color', 'blue');

hold on;
yyaxis right
h1 = plot(xx(1), yy(1), '*', 'LineWidth', 3);
set(h1, 'Color', 'red');

legend({'Ratio of missclassifications','Validation misclassifications', 'MSE', 'Validation MSE'});


%% Subsection 2.3: Subsampling 20% and 80% from classA

% Dividing class A into negative and positive
classAneg = classA(:,classA(1,:) < 0);
classApos = classA(:,classA(1,:) > 0);

% 20% training data from class Aneg
indexA20 = datasample(1:size(classAneg,2), 0.20*size(classAneg,2), 'Replace', false);
% 80% training data from class Apos
indexA80 = datasample(1:size(classApos,2), 0.80*size(classApos,2), 'Replace', false);

% Validation set
valid20    = classAneg(:,indexA20);
valid80    = classApos(:,indexA80);
valid_data = [valid20, valid80];

% Training set
train20 = classAneg;
train80 = classApos;

train20(:,indexA20) = [];
train80(:,indexA80) = [];

data = [train20, train80, classB];

% Training the neural network
[MSE_vec, misclass_vec, W, V] = MLPbackprop(data, hidden_nodes, max_epochs,eta,true,seed,alpha);

% Select validation set and compute errors
[MSE, misclass] = validation(W, V, valid_data);

valid_MSE = MSE;
train_MSE = MSE_vec(end);

valid_misclass = misclass;
train_misclass = misclass_vec(end);

% Plotting
errorplot(1:max_epochs, MSE_vec,misclass_vec,'20% from neg x1 class A 80% from pos x1 class A','Number of epochs')
hold on

xx = [max_epochs, max_epochs];
yy = [valid_MSE, valid_misclass];
h2 = plot(xx(2), yy(2), '*', 'LineWidth', 3);
set(h2, 'Color', 'blue');

hold on;
yyaxis right
ylim([0, 2]);
h1 = plot(xx(1), yy(1), '*', 'LineWidth', 3);
set(h1, 'Color', 'red');

legend({'Ratio of missclassifications','Validation misclassifications', 'MSE', 'Validation MSE'});




%% For loop: hidden node comparison - 25% from each class

% 25% training data from each class
% Generation of random indexes
indexA25 = datasample(1:100, 25, 'Replace', false);
indexB25 = datasample(1:100, 25, 'Replace', false);

% Validation set
classA25 = classA(:,indexA25);
classB25 = classB(:,indexB25);

% Training set
classA75 = classA;
classA75(:,indexA25) = [];
classB75 = classB;
classB75(:,indexB25) = [];

data = [classA75, classB75];
data25 = [classA25, classB25];

% Let's start iterating over a different number of nodes

nodes_vec      = 2:10;
train_MSE      = [];
train_misclass = [];

valid_MSE      = [];
valid_misclass = [];


for node = nodes_vec
    [MSE_vec, misclass_vec, W, V] = MLPbackprop(data, node, max_epochs,eta,true,seed,alpha);
     train_MSE       = [train_MSE, MSE_vec(end)];
     train_misclass  = [train_misclass, misclass_vec(end)];
     [MSE, misclass] = validation(W, V, data25);

     valid_MSE      = [valid_MSE, MSE];
     valid_misclass = [valid_misclass, misclass];

end

% Plotting

errorplot(nodes_vec, train_MSE, train_misclass, '25% from each class - Errors vs Number of nodes', 'Number of hidden nodes');
hold on;

xx = nodes_vec;
yy = [valid_MSE; valid_misclass];
h4 = plot(xx, yy(2,:), '--', 'LineWidth', 1);
set(h4, 'Color', 'black');
hold on;
h2 = plot(xx, yy(2,:), '*', 'LineWidth', 3);
set(h2, 'Color', 'blue');

hold on;
yyaxis right
ylim([0, 2]);
h3 = plot(xx, yy(1,:), '--', 'LineWidth', 1);
set(h3, 'Color', 'black');
hold on;
h1 = plot(xx, yy(1,:), '*', 'LineWidth', 3);
set(h1, 'Color', 'red');

legend({'Ratio of missclassifications', '', 'Validation misclassifications', 'MSE', '', 'Validation MSE'});




%% For loop: hidden node comparison - 50% from A

% 50% training data from each class
% Generation of random indexes
indexA50 = datasample(1:100, 50, 'Replace', false);

% Validation set
classA50 = classA(:,indexA50);

% Training set
classAtrain = classA;
classAtrain(:,indexA50) = [];
classBtrain = classB;

data   = [classAtrain, classBtrain];
data50 = classA50;

% Let's start iterating over a different number of nodes

nodes_vec      = 2:10;
train_MSE      = [];
train_misclass = [];

valid_MSE      = [];
valid_misclass = [];


for node = nodes_vec
    [MSE_vec, misclass_vec, W, V] = MLPbackprop(data, node, max_epochs,eta,true,seed,alpha);
     train_MSE       = [train_MSE, MSE_vec(end)];
     train_misclass  = [train_misclass, misclass_vec(end)];
     [MSE, misclass] = validation(W, V, data50);

     valid_MSE      = [valid_MSE, MSE];
     valid_misclass = [valid_misclass, misclass];

end

% Plotting

errorplot(nodes_vec, train_MSE, train_misclass, '50% from class A - Errors vs Number of nodes', 'Number of hidden nodes');
hold on;

xx = nodes_vec;
yy = [valid_MSE; valid_misclass];
h4 = plot(xx, yy(2,:), '--', 'LineWidth', 1);
set(h4, 'Color', 'black');
hold on;
h2 = plot(xx, yy(2,:), '*', 'LineWidth', 3);
set(h2, 'Color', 'blue');

hold on;
yyaxis right
ylim([0, 3]);
h3 = plot(xx, yy(1,:), '--', 'LineWidth', 1);
set(h3, 'Color', 'black');
hold on;
h1 = plot(xx, yy(1,:), '*', 'LineWidth', 3);
set(h1, 'Color', 'red');

legend({'Ratio of missclassifications', '', 'Validation misclassifications', 'MSE', '', 'Validation MSE'});




%% For loop: hidden node comparison - 20% and 80% from A

% Dividing class A into negative and positive
classAneg = classA(:,classA(1,:) < 0);
classApos = classA(:,classA(1,:) > 0);

% 20% training data from class Aneg
indexA20 = datasample(1:size(classAneg,2), 0.20*size(classAneg,2), 'Replace', false);
% 80% training data from class Apos
indexA80 = datasample(1:size(classApos,2), 0.80*size(classApos,2), 'Replace', false);

% Validation set
valid20    = classAneg(:,indexA20);
valid80    = classApos(:,indexA80);
valid_data = [valid20, valid80];

% Training set
train20 = classAneg;
train80 = classApos;

train20(:,indexA20) = [];
train80(:,indexA80) = [];

data = [train20, train80, classB];


% Let's start iterating over a different number of nodes

nodes_vec      = 2:10;
train_MSE      = [];
train_misclass = [];

valid_MSE      = [];
valid_misclass = [];


for node = nodes_vec
    [MSE_vec, misclass_vec, W, V] = MLPbackprop(data, node, max_epochs,eta,true,seed,alpha);
     train_MSE       = [train_MSE, MSE_vec(end)];
     train_misclass  = [train_misclass, misclass_vec(end)];
     [MSE, misclass] = validation(W, V, valid_data);

     valid_MSE      = [valid_MSE, MSE];
     valid_misclass = [valid_misclass, misclass];

end

% Plotting

errorplot(nodes_vec, train_MSE, train_misclass, '20% and 80% from class A - Errors vs Number of nodes', 'Number of hidden nodes');
hold on;

xx = nodes_vec;
yy = [valid_MSE; valid_misclass];
h4 = plot(xx, yy(2,:), '--', 'LineWidth', 1);
set(h4, 'Color', 'black');
hold on;
h2 = plot(xx, yy(2,:), '*', 'LineWidth', 3);
set(h2, 'Color', 'blue');

hold on;
yyaxis right
ylim([0, 10]);
h3 = plot(xx, yy(1,:), '--', 'LineWidth', 1);
set(h3, 'Color', 'black');
hold on;
h1 = plot(xx, yy(1,:), '*', 'LineWidth', 3);
set(h1, 'Color', 'red');

legend({'Ratio of missclassifications', '', 'Validation misclassifications', 'MSE', '', 'Validation MSE'});





