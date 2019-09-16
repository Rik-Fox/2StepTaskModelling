######### ASκ THE ENVIRON #################

####### DAW
function runSim(n::Int64, α::T, β1::T, β2::T, λ::T, ηₜ::T, ηᵣ::T) where T <: Float64

    epoch_C = zeros(6, n)
    epoch_TV = zeros(6, n)
    epoch = zeros(6, n)
    epochT = zeros(6, n)
    epoch_R = zeros(6, n)
    epoch_e = zeros(6, n)
    Rwds = zeros(n)
    p1 = zeros(n)
    p2 = zeros(n)
    agent = buildAgent(2)
    for i = 1:n

        X = agentCtrller(agent, α, β1, β2, λ, ηₜ, ηᵣ)

        Rwds[i], p1[i], p2[i] = X[2], X[4], X[3][4]

        epoch_C[:,i] = [X[5].state.Q.A1, X[5].state.Q.A2, X[5].μ.state.Q.A1, X[5].μ.state.Q.A2, X[5].ν.state.Q.A1, X[5].ν.state.Q.A2]
        epoch_TV[:,i] = [X[6].state.Q.A1, X[6].state.Q.A2, X[6].μ.state.Q.A1, X[6].μ.state.Q.A2, X[6].ν.state.Q.A1, X[6].ν.state.Q.A2]

        agent = compareVar(X[1],X[5],X[6],epoch_C[:,1:i],epoch_TV[:,1:i])

        epoch[:,i] = [agent.state.Q.A1, agent.state.Q.A2, agent.μ.state.Q.A1, agent.μ.state.Q.A2, agent.ν.state.Q.A1, agent.ν.state.Q.A2]
        epochT[:,i] = [agent.state.T.A1, agent.state.T.A2, agent.μ.state.T.A1, agent.μ.state.T.A2, agent.ν.state.T.A1, agent.ν.state.T.A2]
        epoch_R[:,i] = [agent.state.R.A1, agent.state.R.A2, agent.μ.state.R.A1, agent.μ.state.R.A2, agent.ν.state.R.A1, agent.ν.state.R.A2]
        epoch_e[:,i] = [agent.state.e.A1, agent.state.e.A2, agent.μ.state.e.A1, agent.μ.state.e.A2, agent.ν.state.e.A1, agent.ν.state.e.A2]
    end

    return agent, (Rwds, p1, p2), (epoch, epochT, epoch_R, epoch_e), Nothing()
end

# ####### Dezfouli
# function runSim(n::Int64, α1::T, α2::T, β1::T, β2::T, λ::T, ηₜ::T, κ::T, w::T) where T <: Float64
#
#     # pre allocating data arrays
#     epoch_Q = zeros(6, n)
#     epoch_T = zeros(6, n)
#     epoch_e = zeros(6, n)
#     p1 = zeros(n)
#     p2 = zeros(n)
#     agent = buildAgent(2)
#     prev_actn = ["A1", "A1"]
#     κ
#
#     for i = 1:n
#         X = agentCtrller(agent, α1, α2, β1, β2, λ, ηₜ, κ, prev_actn, w)
#         p1[i], p2[i] = X[4], X[3][4]
#         prev_actn = [X[3][1], X[3][3]]
#
#         epoch_Q[:,i] = [agent.state.Q.A1, agent.state.Q.A2, agent.μ.state.Q.A1, agent.μ.state.Q.A2, agent.ν.state.Q.A1, agent.ν.state.Q.A2]
#         epoch_T[:,i] = [agent.state.T.A1, agent.state.T.A2, agent.μ.state.T.A1, agent.μ.state.T.A2, agent.ν.state.T.A1, agent.ν.state.T.A2]
#         epoch_e[:,i] = [agent.state.e.A1, agent.state.e.A2, agent.μ.state.e.A1, agent.μ.state.e.A2, agent.ν.state.e.A1, agent.ν.state.e.A2]
#     end
#
#     return agent, (Nothing(), p1, p2), (epoch_Q, epoch_T, Nothing(), epoch_e), Nothing()
# end

###### Miller
function runSim(n::Int64, α::T, β1::T, β2::T, ηₜ::T, ηᵣ::T, wₒ::T, wₕ::T, wᵧ::T) where T <: Float64

    # pre allocating data arrays
    epoch_Q = zeros(6, n)
    epoch_T = zeros(6, n)
    epoch_R = zeros(6, n)
    epoch_e = zeros(6, n)
    epoch_H = zeros(6, n)
    w = zeros(n)
    Rwd = zeros(n)
    p1 = zeros(n)
    p2 = zeros(n)
    agent = buildAgent(2, habit = true)

    ## Param init conditions
    D = zeros(6) #runing average of value free weighting on all actions
    β = 1.0

    for i = 1:n

        X = agentCtrller(agent, α, β, ηₜ, ηᵣ, D)
        Rwd[i], p1[i], p2[i] = X[2], X[4], X[3][4]


        π = [softMax([D[1], D[2]], β = β), softMax([D[2], D[1]], β = β), softMax([D[3], D[4]], β = β), softMax([D[4], D[3]], β = β), softMax([D[5], D[6]], β = β), softMax([D[6], D[5]], β = β)]

        epoch_Q[:,i] = [agent.state.Q.A1, agent.state.Q.A2, agent.μ.state.Q.A1, agent.μ.state.Q.A2, agent.ν.state.Q.A1, agent.ν.state.Q.A2]
        epoch_T[:,i] = [agent.state.T.A1, agent.state.T.A2, agent.μ.state.T.A1, agent.μ.state.T.A2, agent.ν.state.T.A1, agent.ν.state.T.A2]
        epoch_R[:,i] = [agent.state.R.A1, agent.state.R.A2, agent.μ.state.R.A1, agent.μ.state.R.A2, agent.ν.state.R.A1, agent.ν.state.R.A2]
        epoch_e[:,i] = [agent.state.e.A1, agent.state.e.A2, agent.μ.state.e.A1, agent.μ.state.e.A2, agent.ν.state.e.A1, agent.ν.state.e.A2]
        epoch_H[:,i] = [agent.state.h.A1, agent.state.h.A2, agent.μ.state.h.A1, agent.μ.state.h.A2, agent.ν.state.h.A1, agent.ν.state.h.A2]

        hσ = abs(sum(epoch_H[:,i] .- (sum(epoch_H[:,1:i-1], dims = 2) ./i)))
        gσ = sqrt(sum(π .*(epoch_R[:,i] .- sum(π .*epoch_R[:,i])).^2 ))

        w[i] = 1 / ( 1 + exp((wᵧ * gσ) - (wₕ * hσ) + wₒ) )
        D = w[i].*β1.*epoch_H[:,i] .+ (1-w[i]).*β2.*epoch_Q[:,i]
    end

    return agent, (Rwd, p1, p2), (epoch_Q, epoch_T, epoch_R, epoch_e, epoch_H), w
end

####### ASκ THE DATA #####################

###### DAW
function runSim(data::Array{Bool,2}, α::T, β1::T, β2::T, λ::T, ηₜ::T, ηᵣ::T) where T <: Float64

    n = length(data[:,1])
    epoch_C = zeros(6, n)
    epoch_TV = zeros(6, n)
    epoch = zeros(6, n)
    epochT = zeros(6, n)
    epoch_R = zeros(6, n)
    epoch_e = zeros(6, n)
    Rwds = zeros(n)
    p1 = zeros(n)
    p2 = zeros(n)
    agent = buildAgent(2)
    for i = 1:n
        X = agentCtrller(agent, α, β1, β2, λ, ηₜ, ηᵣ, data[i,:])

        Rwds[i], p1[i], p2[i] = X[2], X[4], X[3][4]

        epoch_C[:,i] = [X[5].state.Q.A1, X[5].state.Q.A2, X[5].μ.state.Q.A1, X[5].μ.state.Q.A2, X[5].ν.state.Q.A1, X[5].ν.state.Q.A2]
        epoch_TV[:,i] = [X[6].state.Q.A1, X[6].state.Q.A2, X[6].μ.state.Q.A1, X[6].μ.state.Q.A2, X[6].ν.state.Q.A1, X[6].ν.state.Q.A2]

        agent = compareVar(X[1],X[5],X[6],epoch_C[:,1:i],epoch_TV[:,1:i])

        epoch[:,i] = [agent.state.Q.A1, agent.state.Q.A2, agent.μ.state.Q.A1, agent.μ.state.Q.A2, agent.ν.state.Q.A1, agent.ν.state.Q.A2]
        epochT[:,i] = [agent.state.T.A1, agent.state.T.A2, agent.μ.state.T.A1, agent.μ.state.T.A2, agent.ν.state.T.A1, agent.ν.state.T.A2]
        epoch_R[:,i] = [agent.state.R.A1, agent.state.R.A2, agent.μ.state.R.A1, agent.μ.state.R.A2, agent.ν.state.R.A1, agent.ν.state.R.A2]
        epoch_e[:,i] = [agent.state.e.A1, agent.state.e.A2, agent.μ.state.e.A1, agent.μ.state.e.A2, agent.ν.state.e.A1, agent.ν.state.e.A2]
    end

    return agent, (Rwds, p1, p2), (epoch, epochT, epoch_R, epoch_e), Nothing()
end

# ####### DB
# function runSim(data::Array{Bool,2}, α1::T, α2::T, β1::T, β2::T, λ::T, ηₜ::T, κ::T, w::T) where T <: Float64
#     n = length(data[:,1])
#     # pre allocating data arrays
#     epoch_Q = zeros(6, n)
#     epoch_T = zeros(6, n)
#     epoch_e = zeros(6, n)
#     p1 = zeros(n)
#     p2 = zeros(n)
#     agent = buildAgent(2)
#     prev_actn = ["A1", "A1"]
#
#     for i = 1:n
#
#         X = agentCtrller(agent, α1, α2, β1, β2, λ, ηₜ, κ, prev_actn, w, data[i,:])
#         p1[i], p2[i] = X[4], X[3][4]
#         prev_actn = [X[3][1], X[3][3]]
#
#         epoch_Q[:,i] = [agent.state.Q.A1, agent.state.Q.A2, agent.μ.state.Q.A1, agent.μ.state.Q.A2, agent.ν.state.Q.A1, agent.ν.state.Q.A2]
#         epoch_T[:,i] = [agent.state.T.A1, agent.state.T.A2, agent.μ.state.T.A1, agent.μ.state.T.A2, agent.ν.state.T.A1, agent.ν.state.T.A2]
#         epoch_e[:,i] = [agent.state.e.A1, agent.state.e.A2, agent.μ.state.e.A1, agent.μ.state.e.A2, agent.ν.state.e.A1, agent.ν.state.e.A2]
#     end
#
#     return agent, (Nothing(), p1, p2), (epoch_Q, epoch_T, Nothing(), epoch_e), Nothing()
# end

##### Miller
function runSim(data::Array{Bool,2}, α::T, β1::T, β2::T, ηₜ::T, ηᵣ::T, wₒ::T, wₕ::T, wᵧ::T) where T <: Float64

    n = length(data[:,1])
    # pre allocating data arrays
    epoch_Q = zeros(6, n)
    epoch_T = zeros(6, n)
    epoch_R = zeros(6, n)
    epoch_e = zeros(6, n)
    epoch_H = zeros(6, n)
    w = zeros(n)
    Rwd = zeros(n)
    p1 = zeros(n)
    p2 = zeros(n)
    agent = buildAgent(2, habit = true)

    ## Param init conditions
    D = zeros(6)       #runing average of value free weighting on all actions
    β = 1.0

    for i = 1:n
        X = agentCtrller(agent, α, β, ηₜ, ηᵣ, D, data[i,:])
        Rwd[i], p1[i], p2[i] = X[2], X[4], X[3][4]


        π = [softMax([D[1], D[2]], β = β), softMax([D[2], D[1]], β = β), softMax([D[3], D[4]], β = β), softMax([D[4], D[3]], β = β), softMax([D[5], D[6]], β = β), softMax([D[6], D[5]], β = β)]

        epoch_Q[:,i] = [agent.state.Q.A1, agent.state.Q.A2, agent.μ.state.Q.A1, agent.μ.state.Q.A2, agent.ν.state.Q.A1, agent.ν.state.Q.A2]
        epoch_T[:,i] = [agent.state.T.A1, agent.state.T.A2, agent.μ.state.T.A1, agent.μ.state.T.A2, agent.ν.state.T.A1, agent.ν.state.T.A2]
        epoch_R[:,i] = [agent.state.R.A1, agent.state.R.A2, agent.μ.state.R.A1, agent.μ.state.R.A2, agent.ν.state.R.A1, agent.ν.state.R.A2]
        epoch_e[:,i] = [agent.state.e.A1, agent.state.e.A2, agent.μ.state.e.A1, agent.μ.state.e.A2, agent.ν.state.e.A1, agent.ν.state.e.A2]
        epoch_H[:,i] = [agent.state.h.A1, agent.state.h.A2, agent.μ.state.h.A1, agent.μ.state.h.A2, agent.ν.state.h.A1, agent.ν.state.h.A2]

        hσ = abs(sum(epoch_H[:,i] .- (sum(epoch_H[:,1:i-1], dims = 2) ./i) ))
        gσ = sqrt(sum(π .*(epoch_R[:,i] .- sum(π .*epoch_R[:,i])).^2 ))

        w[i] = 1 / ( 1 + exp((wᵧ * gσ) - (wₕ * hσ) + wₒ) )
        D = w[i].*β1.*epoch_H[:,i] .+ (1-w[i]).*β2.*epoch_Q[:,i]
    end

    return agent, (Rwd, p1, p2), (epoch_Q, epoch_T, epoch_R, epoch_e, epoch_H), w
end
