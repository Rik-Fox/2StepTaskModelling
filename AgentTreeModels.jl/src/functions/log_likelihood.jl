using DelimitedFiles

function log_likelihood(θ::Array{Float64,1})

    # D = groupby( DataFrame( CSV.File("/home/rfox/Project_MSc/data/Subj43.csv", delim=',') ), :Flex0_or_Spec1 )[1]
    # data = [t==1 for t in [D.First_Choice [t==0.3 for t in D.Transition_Prob] D.Second_Choice D.Reward]]

    if length(θ) == 6

        data = Array{Bool,2}(readdlm("Data/data_daw.csv"))
        X = runSim(data, θ[1], θ[2], θ[3], θ[4], θ[5], θ[6])

    elseif length(θ) == 8

        data = Array{Bool,2}(readdlm("Data/data_mil.csv"))

        X = runSim(data, θ[1], θ[2], θ[3], θ[4], θ[5], θ[6], θ[7], θ[8])

    # elseif length(θ) == 7
    #
    #     α, β, λ, ηₜ, ηᵣ, M, w, w1, w2 = θ[1], θ[2], θ[3], θ[4], θ[5], θ[6], θ[7], θ[8], θ[9]
    #     X = runSim(data, α, β, λ, ηₜ, ηᵣ, M, w, w1, w2)

    else
        throw(ArgumentError("Number of parameters θ does not match any current model"))
    end

    L = BigFloat(0.0)
    a1_prob1 = convert(Array{BigFloat,1},X[2][2])
    a1_prob2 = convert(Array{BigFloat,1},X[2][3])
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
    return -convert(Float64,L) ## BIGFLOAT??
end
