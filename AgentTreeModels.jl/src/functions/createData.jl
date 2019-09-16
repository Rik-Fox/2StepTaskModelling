#### DAW
function createData(n::Int64, α::T, β1::T, β2::T, λ::T, ηₜ::T, ηᵣ::T) where T <: Float64
    dat = Array{Any,2}(undef, (n, 4))
    agent = buildAgent(2)
    for i = 1:n
        x = agentCtrller(agent, α, β1, β2, λ, ηₜ, ηᵣ)

        dat[i,:] = [x[3][1],x[3][2],x[3][3],x[2]]
    end
    Data = [[d == "A1" for d in dat[:,1]] [d for d in dat[:,2]] [d == "A1" for d in dat[:,3]] [d == 1.0 for d in dat[:,4]]]

        for i = 1:n
            if Data[i,1]
                if Data[i,2]
                    Data[i,2] = false
                else
                    Data[i,2] = true
                end
            end
        end

    return Data
end

### Dezfouli

# function createData(N::Int64, α1::T, α2::T, β1::T, β2::T, λ::T, ηₜ::T, κ::T, w::T) where T <: Float64
#     dat = Array{Any,2}(undef, (N, 4))
#     agent = buildAgent(2)
#     prev_actn = ["A1", "A1"]
#
#     for i = 1:N
#         x = agentCtrller(agent, α1, α2, β1, β2, λ, ηₜ, κ, prev_actn, w)
#
#         dat[i,:] = [x[3][1],x[3][2],x[3][3],x[2]]
#         prev_actn = [dat[i,1], dat[i,3]]
#     end
#     Data = [[d == "A1" for d in dat[:,1]] [d for d in dat[:,2]] [d == "A1" for d in dat[:,3]] [d == 1.0 for d in dat[:,4]]]
#
#         for i = 1:N
#             if Data[i,1]
#                 if Data[i,2]
#                     Data[i,2] = false
#                 else
#                     Data[i,2] = true
#                 end
#             end
#         end
#
#     return Data
# end

### Miller

function createData(n::Int64, α::T, β1::T, β2::T, ηₜ::T, ηᵣ::T, wₒ::T, wₕ::T, wᵧ::T) where T <: Float64
    dat = Array{Any,2}(undef, (n, 4))
    epoch_Q = zeros(6, n)
    epoch_T = zeros(6, n)
    epoch_R = zeros(6, n)
    epoch_e = zeros(6, n)
    epoch_H = zeros(6, n)
    epoch_D = zeros(6, n)
    Rwd = zeros(n)
    p1 = zeros(n)
    p2 = zeros(n)
    agent = buildAgent(2, habit = true)

    ## Param init conditions
    D = zeros(6) #runing average of value free weighting on all actions
    β = 1.0

    for i = 1:n

        x = agentCtrller(agent, α, β, ηₜ, ηᵣ, D)
        Rwd[i], p1[i], p2[i] = x[2], x[4], x[3][4]


        π = [softMax([D[1], D[2]], β = β), softMax([D[2], D[1]], β = β), softMax([D[3], D[4]], β = β), softMax([D[4], D[3]], β = β), softMax([D[5], D[6]], β = β), softMax([D[6], D[5]], β = β)]

        epoch_Q[:,i] = [x[1].state.Q.A1, x[1].state.Q.A2, x[1].μ.state.Q.A1, x[1].μ.state.Q.A2, x[1].ν.state.Q.A1, x[1].ν.state.Q.A2]
        epoch_T[:,i] = [x[1].state.T.A1, x[1].state.T.A2, x[1].μ.state.T.A1, x[1].μ.state.T.A2, x[1].ν.state.T.A1, x[1].ν.state.T.A2]
        epoch_R[:,i] = [x[1].state.R.A1, x[1].state.R.A2, x[1].μ.state.R.A1, x[1].μ.state.R.A2, x[1].ν.state.R.A1, x[1].ν.state.R.A2]
        epoch_e[:,i] = [x[1].state.e.A1, x[1].state.e.A2, x[1].μ.state.e.A1, x[1].μ.state.e.A2, x[1].ν.state.e.A1, x[1].ν.state.e.A2]
        epoch_H[:,i] = [x[1].state.h.A1, x[1].state.h.A2, x[1].μ.state.h.A1, x[1].μ.state.h.A2, x[1].ν.state.h.A1, x[1].ν.state.h.A2]
        epoch_D[:,i] = D

        hσ = abs(sum(epoch_H[:,i] .- (sum(epoch_H[:,1:i-1], dims = 2) ./i)))
        gσ = sqrt(sum(π .*(epoch_R[:,i] .- sum(π .*epoch_R[:,i])).^2 ))

        w = 1 / ( 1 + exp((wᵧ * gσ) - (wₕ * hσ) + wₒ) )
        D = w.*β1.*epoch_H[:,i] .+ (1-w).*β2.*epoch_Q[:,i]

        dat[i,:] = [x[3][1],x[3][2],x[3][3],x[2]]

    end
    Data = [[d == "A1" for d in dat[:,1]] [d for d in dat[:,2]] [d == "A1" for d in dat[:,3]] [d == 1 for d in dat[:,4]]]
    for i = 1:n
        if Data[i,1]
            if Data[i,2]
                Data[i,2] = false
            else
                Data[i,2] = true
            end
        end
    end
    return Data
end
