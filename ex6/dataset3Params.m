function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
L_C = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
L_sigma = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

F_C = 0.01;
F_sigma = 0.01;
F_m = 1;
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
fprintf('Find Optimal C & sigma \n');
for i = 1:size(L_C,2),
  #fprintf('%f',i);
  
  for j = 1:size(L_sigma,2),
    #fprintf('%f',j);
    fprintf('%f, %f trial C & sigma : \n',i,j);
    C = L_C(1,i)
    sigma = L_sigma(1,j)
    model= svmTrain(Xval, yval, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    predictions = svmPredict(model, Xval);
    m =  mean(double(predictions ~= yval))    
    if m <= F_m
      F_m = m;
      F_C = C;
      F_sigma = sigma;
    end
  end
  
end

fprintf('Result of Optimal C & sigma \n');
C = F_C
sigma = F_sigma

%{
model= svmTrain(Xval, yval, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
predictions = svmPredict(model, Xval);
mean(double(predictions ~= yval));
}%

% =========================================================================
end