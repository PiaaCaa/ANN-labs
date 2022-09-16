function [MSE, misclass] = validation(W, V, validation_data)
    % validation compiutes the errors (MSE and misclassification ratio)
    % done by the trained neural network on a salected validation set.
    % Please insert in the third row of validation_data the target labels.

ndata    = size(validation_data, 2);
targets  = validation_data(3,:);
patterns = validation_data(1:2,:);
 

% Forward pass
hin  = W * [patterns; ones(1,ndata)];
hout = [2 ./ (1+exp(-hin)) - 1 ; ones(1,ndata)];
oin  = V * hout;
out  = 2 ./ (1+exp(-oin)) - 1;

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

% Misclassification error
misclass = sum(not(pred==targets))/ndata;

end