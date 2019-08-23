##Miller

function runSim_mixed(agent::DecisionTree, data::Array{Bool,2}, wₒ::T, wₕ::T, wᵧ::T; N::Int=100, α::T=0.5, β::T=5.0, λ::T=0.0, ηₜ::T=0.5, ηᵣ::T=0.5, ϵ_cut::T=0.1) where {T<:Float64, T1<:Actions, T2<:Nothing}
    n = length(data[:,1])
    # pre allocating data arrays
    epoch_Q = zeros(6,n)
    epoch_T = zeros(6,n)
    epoch_H = zeros(6,n)
    Rwd = zeros(n)
    p1 = zeros(n)
    p2 = zeros(n)
    actn1 = Array{AbstractString,1}(undef,n)
    actn2 = Array{AbstractString,1}(undef,n)

    #HWV.state.T.A1, HWV.state.T.A2, HWV.A1.state.T.A1, HWV.A1.state.T.A2, HWV.A2.state.T.A1, HWV.A2.state.T.A2 = 0.7, 0.7, 1.0, 1.0, 1.0, 1.0
    ## Param init conditions
    Pᵧ = ones(n)
    rᵧ = 0.0
    rₒ = 0.0
    H = 0.0         #runing average of value free weighting on all actions

    for i=1:n
        Pᵧ[i] = 1/( 1 + exp( (wᵧ*abs(rᵧ - rₒ)) + (wₕ*abs(H^2) + wₒ) ) )

        if rand() < Pᵧ[i]
            X = agentCtrller_mixed(agent, data[i,:], ϵ_cut, α=α, β=β, λ=λ, ηₜ=ηₜ, ηᵣ=ηᵣ)
            agent = X[1]
            Rwd[i],p1[i],p2[i],actn1[i],actn2[i] = X[2],X[4],X[3][4],X[3][1],X[3][3]
            rᵧ = (1-α)*rᵧ + α*Rwd[i]
        else
            X = agentCtrller_mixed(agent, data[i,:], α=α, β=β, λ=λ, ηₜ=ηₜ, ηᵣ=ηᵣ)
            agent = X[1]
            Rwd[i],p1[i],p2[i],actn1[i],actn2[i] = X[2],X[4],X[3][4],X[3][1],X[3][3]
            H = sum([agent.state.h.A1, agent.state.h.A2, agent.μ.state.h.A1, agent.μ.state.h.A2, agent.ν.state.h.A1, agent.ν.state.h.A2])/6
        end
        rₒ = (1-α)*rₒ + α*Rwd[i]
        epoch_Q[:,i] = [agent.state.Q.A1, agent.state.Q.A2, agent.μ.state.Q.A1, agent.μ.state.Q.A2, agent.ν.state.Q.A1, agent.ν.state.Q.A2]
        epoch_T[:,i] = [agent.state.T.A1, agent.state.T.A2, agent.μ.state.T.A1, agent.μ.state.T.A2, agent.ν.state.T.A1, agent.ν.state.T.A2]
        epoch_H[:,i] = [agent.state.h.A1, agent.state.h.A2, agent.μ.state.h.A1, agent.μ.state.h.A2, agent.ν.state.h.A1, agent.ν.state.h.A2]
    end

    return agent, (Rwd, p1, p2, actn1, actn2), (epoch_Q, epoch_T, epoch_H), Pᵧ
end

function runSim_mixed(agent::DecisionTree, n::Int64, wₒ::T, wₕ::T, wᵧ::T;  α::T=0.5, β::T=5.0, λ::T=0.0, ηₜ::T=0.5, ηᵣ::T=0.5, ϵ_cut::T=0.1) where {T<:Float64, T1<:Actions, T2<:Nothing}
    n = length(data[:,1])
    # pre allocating data arrays
    epoch_Q = zeros(6,n)
    epoch_T = zeros(6,n)
    epoch_H = zeros(6,n)
    Rwd = zeros(n)
    p1 = zeros(n)
    p2 = zeros(n)
    actn1 = Array{AbstractString,1}(undef,n)
    actn2 = Array{AbstractString,1}(undef,n)

    #HWV.state.T.A1, HWV.state.T.A2, HWV.A1.state.T.A1, HWV.A1.state.T.A2, HWV.A2.state.T.A1, HWV.A2.state.T.A2 = 0.7, 0.7, 1.0, 1.0, 1.0, 1.0
    ## Param init conditions
    Pᵧ = ones(n)
    rᵧ = 0.0
    rₒ = 0.0
    H = 0.0         #runing average of value free weighting on all actions

    for i=1:n
        Pᵧ[i] = 1/( 1 + exp( (wᵧ*abs(rᵧ - rₒ)) + (wₕ*abs(H^2) + wₒ) ) )

        if rand() < Pᵧ[i]
            X = agentCtrller_mixed(agent, ϵ_cut, α=α, β=β, λ=λ, ηₜ=ηₜ, ηᵣ=ηᵣ)
            agent = X[1]
            Rwd[i],p1[i],p2[i],actn1[i],actn2[i] = X[2],X[4],X[3][4],X[3][1],X[3][3]
            rᵧ = (1-α)*rᵧ + α*Rwd[i]
        else
            X = agentCtrller_mixed(agent, α=α, β=β, λ=λ, ηₜ=ηₜ, ηᵣ=ηᵣ)
            agent = X[1]
            Rwd[i],p1[i],p2[i],actn1[i],actn2[i] = X[2],X[4],X[3][4],X[3][1],X[3][3]
            H = sum([agent.state.h.A1, agent.state.h.A2, agent.μ.state.h.A1, agent.μ.state.h.A2, agent.ν.state.h.A1, agent.ν.state.h.A2])/6
        end
        rₒ = (1-α)*rₒ + α*Rwd[i]
        epoch_Q[:,i] = [agent.state.Q.A1, agent.state.Q.A2, agent.μ.state.Q.A1, agent.μ.state.Q.A2, agent.ν.state.Q.A1, agent.ν.state.Q.A2]
        epoch_T[:,i] = [agent.state.T.A1, agent.state.T.A2, agent.μ.state.T.A1, agent.μ.state.T.A2, agent.ν.state.T.A1, agent.ν.state.T.A2]
        epoch_H[:,i] = [agent.state.h.A1, agent.state.h.A2, agent.μ.state.h.A1, agent.μ.state.h.A2, agent.ν.state.h.A1, agent.ν.state.h.A2]
    end

    return agent, (Rwd, p1, p2, actn1, actn2), (epoch_Q, epoch_T, epoch_H), Pᵧ
end

####### DB

function runSim_mixed(agent::DecisionTree, data::Array{Bool,2}, w::T; α::T=0.5, β::T=5.0, λ::T=0.0, ηₜ::T=0.5, ηᵣ::T=0.5, ϵ_cut::T=0.1) where {T<:Float64, T1<:Actions, T2<:Nothing}
    n = length(data[:,1])
    # pre allocating data arrays
    epoch_Q = zeros(6,n)
    epoch_T = zeros(6,n)
    Rwd = zeros(n)
    p1 = zeros(n)
    p2 = zeros(n)
    actn1 = Array{AbstractString,1}(undef,n)
    actn2 = Array{AbstractString,1}(undef,n)

    #HWV.state.T.A1, HWV.state.T.A2, HWV.A1.state.T.A1, HWV.A1.state.T.A2, HWV.A2.state.T.A1, HWV.A2.state.T.A2 = 0.7, 0.7, 1.0, 1.0, 1.0, 1.0
    ## Param init conditions
    Pᵧ = ones(n)
    rᵧ = 0.0
    rₒ = 0.0
    H = 0.0

    for i=1:n
        Pᵧ[i] = 1/( 1 + exp( (w*abs(rᵧ - rₒ)) + ((1-w)*abs(H^2)) ) )

        if rand() < Pᵧ[i]
            X = agentCtrller_mixed(agent, data[i,:], ϵ_cut, α=α, β=β, λ=λ, ηₜ=ηₜ, ηᵣ=ηᵣ)
            agent = X[1]
            Rwd[i],p1[i],p2[i],actn1[i],actn2[i] = X[2],X[4],X[3][4],X[3][1],X[3][3]
            rᵧ = (1-α)*rᵧ + α*Rwd[i]
        else
            X = agentCtrller_mixed(agent, data[i,:], α=α, β=β, λ=λ, ηₜ=ηₜ, ηᵣ=ηᵣ)
            agent = X[1]
            Rwd[i],p1[i],p2[i],actn1[i],actn2[i] = X[2],X[4],X[3][4],X[3][1],X[3][3]
            h_avg = sum([agent.state.Q.A1, agent.state.Q.A2, agent.μ.state.Q.A1, agent.μ.state.Q.A2, agent.ν.state.Q.A1, agent.ν.state.Q.A2])/6
        end
        rₒ = (1-α)*rₒ + α*Rwd[i]
        epoch_Q[:,i] = [agent.state.Q.A1, agent.state.Q.A2, agent.μ.state.Q.A1, agent.μ.state.Q.A2, agent.ν.state.Q.A1, agent.ν.state.Q.A2]
        epoch_T[:,i] = [agent.state.T.A1, agent.state.T.A2, agent.μ.state.T.A1, agent.μ.state.T.A2, agent.ν.state.T.A1, agent.ν.state.T.A2]
    end

    return agent, (Rwd, p1, p2, actn1, actn2), (epoch_Q, epoch_T), Pᵧ
end

function runSim_mixed(agent::DecisionTree, n::Int64, w::T; α::T=0.5, β::T=5.0, λ::T=0.0, ηₜ::T=0.5, ηᵣ::T=0.5, ϵ_cut::T=0.1) where {T<:Float64, T1<:Actions, T2<:Nothing}
    # pre allocating data arrays
    epoch_Q = zeros(6,n)
    epoch_T = zeros(6,n)
    Rwd = zeros(n)
    p1 = zeros(n)
    p2 = zeros(n)
    actn1 = Array{AbstractString,1}(undef,n)
    actn2 = Array{AbstractString,1}(undef,n)

    #HWV.state.T.A1, HWV.state.T.A2, HWV.A1.state.T.A1, HWV.A1.state.T.A2, HWV.A2.state.T.A1, HWV.A2.state.T.A2 = 0.7, 0.7, 1.0, 1.0, 1.0, 1.0
    ## Param init conditions
    Pᵧ = ones(n)
    rᵧ = 0.0
    rₒ = 0.0
    H = 0.0

    for i=1:n
        Pᵧ[i] = 1/( 1 + exp( (w*abs(rᵧ - rₒ)) + ((1-w)*abs(H^2)) ) )

        if rand() < Pᵧ[i]
            X = agentCtrller_mixed(agent, ϵ_cut, α=α, β=β, λ=λ, ηₜ=ηₜ, ηᵣ=ηᵣ)
            agent = X[1]
            Rwd[i],p1[i],p2[i],actn1[i],actn2[i] = X[2],X[4],X[3][4],X[3][1],X[3][3]
            rᵧ = (1-α)*rᵧ + α*Rwd[i]
        else
            X = agentCtrller_mixed(agent, α=α, β=β, λ=λ, ηₜ=ηₜ, ηᵣ=ηᵣ)
            agent = X[1]
            Rwd[i],p1[i],p2[i],actn1[i],actn2[i] = X[2],X[4],X[3][4],X[3][1],X[3][3]
            h_avg = sum([agent.state.Q.A1, agent.state.Q.A2, agent.μ.state.Q.A1, agent.μ.state.Q.A2, agent.ν.state.Q.A1, agent.ν.state.Q.A2])/6
        end
        rₒ = (1-α)*rₒ + α*Rwd[i]
        epoch_Q[:,i] = [agent.state.Q.A1, agent.state.Q.A2, agent.μ.state.Q.A1, agent.μ.state.Q.A2, agent.ν.state.Q.A1, agent.ν.state.Q.A2]
        epoch_T[:,i] = [agent.state.T.A1, agent.state.T.A2, agent.μ.state.T.A1, agent.μ.state.T.A2, agent.ν.state.T.A1, agent.ν.state.T.A2]
    end

    return agent, (Rwd, p1, p2, actn1, actn2), (epoch_Q, epoch_T), Pᵧ
end
