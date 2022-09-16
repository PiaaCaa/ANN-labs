function [MSE_vec, misclass_vec, W, V] = MLPbackprop3D(x, y, z, hidden_nodes, max_epochs, eta, batch, seed, alpha)

rng(seed)
epoch = 0;

% Initialize hyperparameters for dynamic plot
gridsize = length(x);
ndata    = gridsize*gridsize;

targets  = reshape (z, 1, ndata);
[xx, yy] = meshgrid (x, y);
patterns = [reshape(xx, 1, ndata); reshape(yy, 1, ndata)];

data     = [patterns; targets];

% Initialize random weights w1, w2, w0 (bias term)
w  = randn(hidden_nodes, size(data,1));
v  = randn(1, hidden_nodes + 1);
dw = zeros(hidden_nodes, size(data,1));
dv = zeros(1, hidden_nodes + 1);

MSE_vec      = [];
misclass_vec = [];

while epoch < max_epochs
    
    % Shuffle if sequential
    if batch == false
        data = data(:, randperm(ndata));
    end

    % Divide the data into patterns and the labels (3rd row in data)
    patterns = data(1:2,:);
    targets = data(3,:);

    % Forward pass
    hin = w * [patterns ; ones(1,ndata)];
    hout = [2 ./ (1+exp(-hin)) - 1 ; ones(1,ndata)];
    oin = v * hout;
    out = 2 ./ (1+exp(-oin)) - 1;

    % Backprop
    delta_o = (out - targets) .* ((1 + out) .* (1 - out)) * 0.5;
    delta_h = (v' * delta_o) .* ((1 + hout) .* (1 - hout)) * 0.5;
    delta_h = delta_h(1:hidden_nodes, :);

    % Weight update
    pat = [patterns ; ones(1,ndata)];
    dw = (dw .* alpha) - (delta_h * pat') .* (1-alpha);
    dv = (dv .* alpha) - (delta_o * hout') .* (1-alpha);
    w = w + dw .* eta;
    v = v + dv .* eta;

    % Prediction
    pred = [];
    for i = 1:ndata
        if out(i) > 0
            pred(i) = 1;
        else
            pred(i) = -1;
        end
    end

    % MSE Error
    MSE = sum((out - targets).^2)/ndata;
    MSE_vec = [MSE_vec, MSE];

    % Misclassification error
    misclass_ratio = sum(not(pred==targets))/ndata;
    misclass_vec   = [misclass_vec,misclass_ratio];

    % Plot the result
    zz = reshape(out, gridsize, gridsize);
    mesh(x,y,zz);
    axis([-5 5 -5 5 -0.7 0.7]);
    drawnow;


    % Increment epoch by one
    epoch = epoch + 1;
end

W = w;
V = v;

end