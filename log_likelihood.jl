using DelimitedFiles

function log_likelihood(θ::Array{Float64,1})

    ## if statement selects correct model using multple dispatch on runSim methods corresponding the parameter array θ
    if length(θ) == 6

        data = Array{Bool,2}(readdlm("Data/data_daw.csv"))
        X = runSim(data, θ[1], θ[2], θ[3], θ[4], θ[5], θ[6])

    elseif length(θ) == 8

        data = Array{Bool,2}(readdlm("Data/data_mil.csv"))

        X = runSim(data, θ[1], θ[2], θ[3], θ[4], θ[5], θ[6], θ[7], θ[8])
        ## after implementing eligibility traces Dez and Mil models have same parameters, so abit hacky but have to comment out the method that isn't wanted in runSim function file
    else
        throw(ArgumentError("Number of parameters θ does not match any current model"))
    end
    ## log function in base can sporadical underflow giving undef error so used BigFloats to avoid
    L = BigFloat(0.0)
    ## pulls the softmax probability of action A1 being taken and each trial and each step X[2][2] is starting choice, X[2][3] is secondary choice
    a1_prob1 = convert(Array{BigFloat,1},X[2][2])
    a1_prob2 = convert(Array{BigFloat,1},X[2][3])
    ## ask if A1 is taken, uses log(p) if true, and log(1-p) if A2 was taken, as convient to calc just for A1 as ℙ(A2)= 1- ℙ(A1) given a binary choice
    for i=1:length(a1_prob1)
        if data[i,1]
            L += log(a1_prob1[i])
        else
            L += log(1-a1_prob1[i])
        end
        if data[i,3]
            L += log(a1_prob2[i])
        else
            L += log(1-a1_prob2[i])
        end
    end

    ## Optim packages need to recieve Float64 outputs
    return -convert(Float64,L)
end
