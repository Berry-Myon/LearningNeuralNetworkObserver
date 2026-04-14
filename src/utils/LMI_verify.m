function [flag,P,T] = LMI_verify(A,N_pix,N_piomega,N_xix,N_xiomega)
    
    A_bar = A + N_pix;
    
    [nP,~] = size(A_bar);
    [nT,~] = size(N_xix);
    
    setlmis([])
    P = lmivar(1, [nP 1]);
    T = lmivar(1, repmat([1, 0], nT, 1));
    
    lmiterm([1,1,1,P],A_bar',1,'s');
    lmiterm([1,2,1,-P],N_piomega',1);
    lmiterm([1,2,1,-T],1,N_xix);
    lmiterm([1,2,2,T],1,N_xiomega,'s');
    lmiterm([1,2,2,T],-2,1);
    
    lmiterm([-2,1,1,P],1,1);
    lmiterm([-3,1,1,T],1,1);
    
    solution = getlmis;
    
    [t_min,b] = feasp(solution);
    
    flag = t_min<0;
    
    if flag
        P = dec2mat(solution,b,P);
        T = dec2mat(solution,b,T);
    else
        P = NaN;
        T = NaN;
    end

end