function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the c,X,ost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

z = sigmoid(X*theta);
pred = -(y.*log(z) + (1-y).*log(1-z));
theta_reg = theta(2:length(theta));
var = sum(theta_reg.^2);
J = (1/m)*sum(pred) + (lambda/(2*m))*var;

pred = (z - y)';
theta_reg = theta;
theta_reg(1,1) = 0;
val1 = (1/m)*(pred*X);
val2 = (lambda/m)*theta_reg;
val2 = val2';
grad = val1 + val2;






% =============================================================

end
