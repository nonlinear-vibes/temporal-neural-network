function [param, v, s] = adamUpdate(param, grad, v, s, beta1, beta2, t, eta, m, eps)
    % m = number of samples (for normalizing the step)

    % Wrap numeric arrays into cell for uniform processing
    isCell = iscell(param);
    if ~isCell
      param = {param}; grad = {grad}; v = {v}; s = {s};
    end

    for i = 1:numel(param)
        v{i} = beta1*v{i} + (1-beta1)*grad{i};
        s{i} = beta2*s{i} + (1-beta2)*(grad{i}.^2);
        v_corr = v{i}/(1 - beta1^t);
        s_corr = s{i}/(1 - beta2^t);
        param{i} = param{i} - (eta/m) * v_corr ./ (sqrt(s_corr) + eps);
    end

    % Unwrap if original input was numeric
    if ~isCell
        param = param{1}; v = v{1}; s = s{1};
    end

end