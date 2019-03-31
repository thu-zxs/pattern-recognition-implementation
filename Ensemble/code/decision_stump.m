function [k, a, d] = decision_stump(X, y, w)
% decision_stump returns a rule ...
% h(x) = d if x(k) ¡Ü a, -d otherwise,
%
% Input
%     X : n * p matrix, each row a sample
%     y : n * 1 vector, each row a label
%     w : n * 1 vector, each row a weight
%
% Output
%     k : the optimal dimension
%     a : the optimal threshold
%     d : the optimal d, 1 or -1

% total time complexity required to be O(p*n*logn) or less

minErr = inf;
[n, p] = size(X);
epsilon = 1e-5;

for dim = 1:p
    [X_cur, index] = sort(X(:, dim));
    y_cur = y(index);
    w_cur = w(index);
    
    for d_cur = [1, -1]
        match = (-d_cur*ones(n, 1)) ~= y_cur;
        
        for j = 0:n
            if j ~= 0
                match(j) = ~match(j);
            end
            if w_cur.'*match < minErr
                minErr = w_cur.'*match;
                k = dim;
                if j == 0
                    a = X_cur(1) - epsilon;
                else
                    a = X_cur(j);
                end
                d = d_cur;
            end
        end
    end
end
end
