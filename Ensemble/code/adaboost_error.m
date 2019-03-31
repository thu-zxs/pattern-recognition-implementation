function e = adaboost_error(X, y, k, a, d, alpha)
% adaboost_error: returns the final error rate of a whole adaboost
% 
% Input
%     X     : n * p matrix, each row a sample
%     y     : n * 1 vector, each row a label
%     k     : iter * 1 vector,  selected dimension of features
%     a     : iter * 1 vector, selected threshold for feature-k
%     d     : iter * 1 vector, 1 or -1
%     alpha : iter * 1 vector, weights of the classifiers
%
% Output
%     e     : error rate      

for i = 1:length(d)
    if d(i) == 0
        n = i-1;
        break;
    end
end

p = zeros(length(y),1);
if d(end) ~= 0
    n = length(d);
end

for i = 1:n
    p = p + 2*alpha(i)*d(i)*((X(:, k(i)) <= a(i))-0.5);
end

p = 2*((p>0)-0.5) ~= y;
e = sum(p)/length(y);

end
