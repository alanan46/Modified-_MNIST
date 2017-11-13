%%%%%%%%%%%%%%%%%%%%%
% COMP 551: Applied Machine Learning
% PROJECT 3: Modified MNIST
% AUTHOR: Victoria Madge
% ID: 260789644
% EMAIL: victoria.madge@mail.mcgill.ca
%%%%%%%%%%%%%%%%%%%%%

close all

% Class distribution
A = csvread('/Users/Victoria/Dropbox/Masters/School/COMP 551/Projects/Project 3/train_y.csv');

class = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 24, 25, 27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 54, 56, 63, 64, 72, 81];
class_distn = zeros(1,length(class));
for i = 1:length(class)
   class_distn(i) = numel(find(A==class(i)));
end
sum(class_distn)

% figure
% pie(class_distn)
class
figure
bar(class,class_distn)
axis([0 85 0 6000])
title('Class Distribution');
xlabel('Class Label');
ylabel('Number of Training Instances');

%  CV results
% % x = [1 2 3 4 5 6 7 8 9 10];
% % y = [10.7 10.75 10.95 11.1 11.5 11.95 11.95 11.95 11.95 11.95];
% % figure; plot(x,y);
% % title('Cross Validation of Hidden Layer Hyper-parameter');
% % xlabel('Number of hidden layers');
% % ylabel('Validation Accuracy (%)');

