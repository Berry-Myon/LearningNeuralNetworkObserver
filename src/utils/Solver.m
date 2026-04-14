function [weights, total_time, found_valid_P] = Solver(A, ~, C, L, epsilon, nodes, max_iterations)    
    [nA, ~] = size(A);
    [nC, ~] = size(C);
    A = [epsilon*A, eye(nA); zeros(nA, nA + nA)];
    C = [C, zeros(nC, nA)];
    assert(Hurwitz(A,C,L));
    found_valid_P = false;
    tic;
    [~,layers] = size(nodes);
    for i = 1:max_iterations
        weights = -0.5 + rand(1,layers+1);
        try
            [N_pix, N_piomega, N_xix, N_xiomega] = NN_paras(A, C, nodes, weights, L*C);
            [~, P, ~] = LMI_verify(A, N_pix, N_piomega, N_xix, N_xiomega);
            if exist('P', 'var') && ~any(isnan(P(:)))
                found_valid_P = true;
                break;
            end
        catch
            continue;
        end
    end
    total_time = toc;
end