function L = Pole_Placement(A, ~, C, poles)
    epsilon = 0.01;
    [nA, ~] = size(A);
    [nC, ~] = size(C);
    A = [epsilon*A, eye(nA); zeros(nA, nA + nA)];
    C = [C, zeros(nC, nA)];
    L = -place(A', C', poles)';
    assert(Hurwitz(A,C,L));
end
