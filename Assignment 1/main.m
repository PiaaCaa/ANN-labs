% Assignment 1
clc
close all

% Global variables
hidden_nodes = 2;
epochs = 40;
learning_rate = 0.001;

% Generate dataset
ndata = 100;
mA = [ 1.0, 0.3]; sigmaA = 0.2;
mB = [ 0.0, -0.1]; sigmaB = 0.3;
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
