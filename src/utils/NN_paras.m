function [N_pix,N_piomega,N_xix,N_xiomega] = NN_paras(A,C,nodes,weights,W_LC)

    [nstate,~] = size(A);
    [nobserver,~] = size(C);
    [~,layers] = size(nodes);
    
    N_pix = W_LC;
    N_piomega = [zeros(nstate,sum(nodes)-nodes(layers)) weights(layers+1)*ones(nstate,nodes(layers))];
    N_xix = [weights(1)*ones(nodes(1),nobserver)*C; zeros(sum(nodes)-nodes(1),nstate)];
    
    N_xiomega = zeros(nodes(1),sum(nodes));
    for i=2:layers
        nL = 0;
        nR = 0;
        for j=1:i-2
            nL = nL + nodes(j);
        end
        for j=i:layers
            nR = nR + nodes(j);
        end
        Wi = [zeros(nodes(i),nL) weights(i)*ones(nodes(i),nodes(i-1)) zeros(nodes(i),nR)];
        N_xiomega = [N_xiomega; Wi];
    end
    
end