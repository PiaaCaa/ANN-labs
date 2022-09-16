%% Create the input vectors x and y and the outputs z, then make a 3D plot

clear all;
close all;
clc;

x = [-5:0.5:5]';
y = [-5:0.5:5]';
z = exp(-x.*x*0.1) * exp(-y.*y*0.1)' - 0.5;
mesh(x, y, z);

%% Train the network and visualise the approximated function

% Set up hyperparameters and create the dataset
hidden_nodes  = 18;
max_epochs    = 300;
eta           = 0.001;
batch         = true;
seed          = 5;
alpha         = 0.9;

[MSE_vec, misclass_vec, W, V] = MLPbackprop3D(x, y, z, hidden_nodes, max_epochs, eta, batch, seed, alpha)


%% Computing the MSE depending on the numer of nodes

% General hyperparameters
max_epochs    = 200;
eta           = 0.001;
batch         = true;
seed          = 5;
alpha         = 0.9;

% Nodes
nodes_vec      = [1, 4, 7, 10, 12, 15, 17, 18, 20, 22, 25];
MSE_nodes      = [];
misclass_nodes = [];

for hidden_nodes = nodes_vec
    [MSE_vec, misclass_vec, W, V] = MLPbackprop3D(x, y, z, hidden_nodes, max_epochs, eta, batch, seed, alpha)
    MSE_nodes      = [MSE_nodes, MSE_vec(end)];
end

% Plot the error
ylim_max = 0.1;
MSEplot(nodes_vec, MSE_nodes, ylim_max, 'MSE and number of hidden nodes', 'Number of hidden nodes');

%% Varying number of the training samples (from 80% down to 20%)
%  i.e. subsampling network training and testing --> we will be dividing
%  the available dataset into training and validation sets

% Set up hyperparameters and create the dataset
hidden_nodes  = 18;
max_epochs    = 200;
eta           = 0.05;
batch         = true;
seed          = 468;
alpha         = 0.9;
train_percent = 0.2:0.05:1;

MSE_train = [];
MSE_valid = [];

for train = train_percent
    [MSE_vec, misclass_vec, W, V, validation_data] = MLPsubsample(x, y, z, hidden_nodes, max_epochs, eta, batch, seed, alpha, train);
    MSE_train = [MSE_train, MSE_vec(end)];
    MSE_valid = [MSE_valid, validation(W, V, validation_data)];
end

% Plot the training MSE for the different sets of subsampled data
ylim_max = 0.01;
MSEplot(train_percent, MSE_train, ylim_max, 'MSE for subsampled data', 'Subsampled data percentages')

% Add validation MSE line to the plot and adjust the legend
hold on;
plot(train_percent, MSE_valid, "blue");
legend({'Training MSE', 'Validation MSE'});





