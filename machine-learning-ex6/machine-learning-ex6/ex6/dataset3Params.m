function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

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

C_vect = [0.01;0.03;0.1;0.3;1;3;10;30];
sigma_vect = [0.01;0.03;0.1;0.3;1;3;10;30];
error = zeros(length(C_vect)*length(sigma_vect),3);

index = 1;
for i=1:length(C_vect),
	for j=1:length(sigma_vect),
		m = C_vect(i,1);
		n = sigma_vect(j,1);
		model= svmTrain(X, y, m, @(x1, x2) gaussianKernel(x1, x2, n)); 
		predictions = svmPredict(model,Xval);
		error(index,1) = C_vect(i,1);
		error(index,2) = sigma_vect(j,1);
		error(index,3) = mean(double(predictions ~= yval));
		index = index + 1;
	end,
end,

[val index] = min(error(:,3));
C = error(index,1);
sigma = error(index,2);





% =========================================================================

end
