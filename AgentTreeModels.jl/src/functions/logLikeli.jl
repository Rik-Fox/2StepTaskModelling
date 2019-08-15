function logLikeli(L, modelFit, data)
    p = modelFit[2],modelFit[3]
    for i=1:length(p[1])
        if data[i,1]
            L += log(p[1][i])
        else
            L += log(1-p[1][i])
        end
        if data[i,3]
            L += log(p[2][i])
        else
            L += log(1p[2][i])
        end
    end
    return L
end
