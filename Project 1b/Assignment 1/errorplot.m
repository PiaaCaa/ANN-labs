function [] = errorplot(xdata, MSE_vec, misclass_vec, title_string, xlabel_string)

figure()

% Plotting MSE line (right y axis)
yyaxis right
plot(xdata, MSE_vec, "red");
ylim([0, 1.5]);
ylabel("MSE");
hold on

% Plotting misclassification ratio line (left y axis)
yyaxis left
plot(xdata, misclass_vec, "blue");
ylim([0, 1]);
ylabel("Misclassification ratio");


% Complete the plot and add labels
title(title_string);
xlabel(xlabel_string);
legend({'Ratio of missclassifications','MSE'});

end