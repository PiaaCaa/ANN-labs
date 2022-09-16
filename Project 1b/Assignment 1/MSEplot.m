function [] = MSEplot(xdata, MSE_vec, ylim_max, title_string, xlabel_string)

figure()

% Plotting MSE line (right y axis)
plot(xdata, MSE_vec, "red");
ylim([0, ylim_max]);
ylabel("MSE");

% Complete the plot and add labels
title(title_string);
xlabel(xlabel_string);
legend({'MSE'});

end