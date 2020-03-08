function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a d
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

% Part 1: Feedforward the neural network and return the cost in the variable J.
%fprintf('\n HEREHEREHEREHEREHERE \n')

%y_col = size(y) / num_labels;
%y = reshape(y, num_labels, y_col);

X = [ones(m,1) X];
ext_Theta1 = [ones(1,size(Theta1, 2)) ; Theta1];

for c = 1:m  
  %1
  x = X(c, :);
  a_1 = x';
  x_lay1 = sigmoid(x * ext_Theta1');
  z_2 = (a_1' * ext_Theta1')';
  a_2 = sigmoid(z_2')';
  h = sigmoid(x_lay1 * Theta2')';
  z_3 = (a_2' * Theta2')';
  a_3 = sigmoid(z_3);
  
  %2
  %y_c = [ 1 ; 2 ; 3 ; 4 ; 5 ; 6 ; 7 ; 8 ; 9 ;10];
  y_c = (1:1:num_labels)';
  y_c = y_c== y(c,1);

  %3
  q_3 = a_3 .- y_c;
  q_2 = Theta2' * q_3 .* sigmoidGradient(z_2);
 
  %4
  Theta1_grad = Theta1_grad + q_2(2:end) * a_1';
  Theta2_grad = Theta2_grad + q_3 * a_2';
 
  J = J + sum((-1 .* y_c .* log(h)) .- ((1 .- y_c) .* log(1 .- h)));
  
endfor

J = 1 / m * J;
temp = lambda / (2*m) * (sum(sum(Theta1(:,2:end) .^ 2,1)) + sum(sum(Theta2(:,2:end) .^ 2, 1)));
J = J + temp;

%5
%add1 = (lambda / m ) .* ones(size(Theta1_grad,1), size(Theta1_grad,2));
%add2 = (lambda / m ) .* ones(size(Theta2_grad,1), size(Theta2_grad,2));
add1 = (lambda / m ) .* Theta1;
add2 = (lambda / m ) .* Theta2;

add1(:,1) = 0;
add2(:,1) = 0;

Theta1_grad = (1 / m) .* Theta1_grad .+ add1;
Theta2_grad = (1 / m) .* Theta2_grad .+ add2;
grad = [Theta1_grad(:) ; Theta2_grad(:)];

% Part 2: Implement the backpropagation algorithm to compute the gradients Theta1_grad and Theta2_grad. 

%delta = zeros(size(

end

