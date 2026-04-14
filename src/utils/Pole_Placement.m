function L = Pole_Placement(A, ~, C, poles)
    epsilon = 0.01;
    [nA, ~] = size(A);
    [nC, ~] = size(C);
    A = [epsilon*A, eye(nA); zeros(nA, nA + nA)];
    C = [C, zeros(nC, nA)];
    L = -place(A', C', poles)';
    assert(Hurwitz(A,C,L));

    mfile_dir = fileparts(mfilename('fullpath'));
    weights_dir = fullfile(mfile_dir, '..', '..', 'Weights');
    if ~exist(weights_dir, 'dir')
        mkdir(weights_dir);
    end
    writetable(table(L), fullfile(weights_dir, 'L.csv'))
end