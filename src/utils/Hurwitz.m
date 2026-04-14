function flag = Hurwitz(A,C,L)
    [n,~] = size(A);
    flag = true;
    a = real(eig(A + L * C));
        for i=1:n
            flag = a(i) < 0;
            if ~flag
                break
            end
        end
end