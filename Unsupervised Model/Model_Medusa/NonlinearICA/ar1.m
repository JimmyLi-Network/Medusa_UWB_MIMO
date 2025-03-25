
function y = ar1(phi, seqlen)

    seq = nan*ones(seqlen, 1);

    y(1) = randn(1);
    for t = 1:seqlen-1
        y(t+1) = phi*y(t) + randn(1);
    end

end


