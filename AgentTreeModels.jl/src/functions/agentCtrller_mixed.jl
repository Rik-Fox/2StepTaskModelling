using Random

### run Epochs of Models comprised of Multiple Controllers ###

## ask the environ - model-free VALUE FREE
function agentCtrller_mixed(agent::DecisionTree{State{String,T1,T1,T1,T1}}; α::T=0.5, β::T=5.0, λ::T=0.0, ηₜ::T=0.5, ηᵣ::T=0.5) where {T<:Float64, T1<:Actions, T2<:Nothing}
    stNm = agent.state.name
    π = softMax([agent.state.h.A1, agent.state.h.A2], β=β)

    if π > rand()
        actn = "A1"
        agent.state.e.A1 = 1.0
        Q_old = agent.state.Q.A1
    else
        actn = "A2"
        agent.state.e.A2 = 1.0
        Q_old = agent.state.Q.A2
    end

    μ, Rwd = askEnviron(stNm, actn)
    if stNm == "ξ"
        agent.state.h = habitUpdate(agent.state.h, actn, α)
        if μ
            SC=agentCtrller_mixed(agent.μ, α=α, β=β, λ=λ, ηₜ=ηₜ, ηᵣ=ηᵣ)
            agent.state.h = habitUpdate(agent.state.h, SC[3], α)
        else
            SC=agentCtrller_mixed(agent.ν, α=α, β=β, λ=λ, ηₜ=ηₜ, ηᵣ=ηᵣ)
            agent.state.h = habitUpdate(agent.state.h, SC[3], α)
        end
        Rwd += SC[2]
        actn = actn, μ, SC[3], SC[4]
    end

    return agent, Rwd, actn, π
end

## ask the environ - model-free
function agentCtrller_mixed(agent::DecisionTree{State{String,T1,T1,T2,T1}}; α::T=0.5, β::T=5.0, λ::T=0.0, ηₜ::T=0.5, ηᵣ::T=0.5) where {T<:Float64, T1<:Actions, T2<:Nothing}
    stNm = agent.state.name
    π = softMax([agent.state.Q.A1, agent.state.Q.A2], β=β)

    if π > rand()
        actn = "A1"
        agent.state.e.A1 = 1.0
    else
        actn = "A2"
        agent.state.e.A2 = 1.0
    end

    μ, Rwd = askEnviron(stNm, actn)

    if stNm == "ξ"
        agentUpdate_mixed(agent, actn, Rwd, μ, α, λ, ηₜ, ηᵣ)
        if μ
            SC=agentCtrller_mixed(agent.μ, α=α, β=β, λ=λ, ηₜ=ηₜ, ηᵣ=ηᵣ)
            agentUpdate_mixed(agent, "μ", SC[3], SC[2], μ, α, λ, ηₜ, ηᵣ)
        else
            SC=agentCtrller_mixed(agent.ν, α=α, β=β, λ=λ, ηₜ=ηₜ, ηᵣ=ηᵣ)
            agentUpdate_mixed(agent, "ν", SC[3], SC[2], μ, α, λ, ηₜ, ηᵣ)
        end
        Rwd += SC[2]
        actn = actn, μ, SC[3], SC[4]
    end

    return agent, Rwd, actn, π
end

## ask the environ - model based
function agentCtrller_mixed(agent::DecisionTree, ϵ_cut::T; α::T=0.5, β::T=5.0, λ::T=0.0, ηₜ::T=0.5, ηᵣ::T=0.5) where {T<:Float64, T1<:Actions, T2<:Nothing}
    stNm = agent.state.name
    π = softMax([agent.state.Q.A1, agent.state.Q.A2], β=β)

    π > rand() ? actn = "A1" : actn = "A2"

    μ, Rwd = askEnviron(stNm, actn)

    if stNm == "ξ"
        agentUpdate_mixed(agent, actn, Rwd, μ, α, λ, ηₜ, ηᵣ)
        if μ
            SC=agentCtrller_mixed(agent.μ, ϵ_cut, α=α, β=β, λ=λ, ηₜ=ηₜ, ηᵣ=ηᵣ)
            agentUpdate_mixed(agent, "μ", SC[3], SC[2], μ, α, λ, ηₜ, ηᵣ)
        else
            SC=agentCtrller_mixed(agent.ν, ϵ_cut, α=α, β=β, λ=λ, ηₜ=ηₜ, ηᵣ=ηᵣ)
            agentUpdate_mixed(agent, "ν", SC[3], SC[2], μ, α, λ, ηₜ, ηᵣ)
        end
        Rwd += SC[2]
        actn = actn, μ, SC[3], SC[4]

        ### VI(Model Based) part
        ϵ = 1.0
        while ϵ > ϵ_cut
            Q_old = agent.state.Q
            # agent update will recursively solve all trajectories
            agentUpdate_mixed(agent, "A1", α)
            agentUpdate_mixed(agent, "A2", α)
            ϵ = Δ(Q_old,agent.state.Q)
        end

    end
    return agent, Rwd, actn, π
end

######## Miller ###############

## ask the data - model free
function agentCtrller_mixed(agent::DecisionTree{State{String,T1,T1,T1,T1}}, data::Array{Bool,1}; root::Bool=true, α::T=0.5, β::T=5.0, λ::T=0.0, ηₜ::T=0.5, ηᵣ::T=0.5) where {T<:Float64, T1<:Actions}
    stNm = agent.state.name
    π = softMax([agent.state.Q.A1, agent.state.Q.A2], β=β)

    if data[1]
        actn = "A1"
        agent.state.e.A1 = 1.0
    else
        actn = "A2"
        agent.state.e.A2 = 1.0
    end

    μ, Rwd = askEnviron(stNm,data[1:2])
    if stNm == "ξ"
        agent.state.h = habitUpdate(agent.state.h, actn, α)
        if μ
            SC=agentCtrller_mixed(agent.μ, data[3:4], α=α, β=β, λ=λ, ηₜ=ηₜ, ηᵣ=ηᵣ)
            agent.state.h = habitUpdate(agent.state.h, SC[3], α)
        else
            SC=agentCtrller_mixed(agent.ν, data[3:4], α=α, β=β, λ=λ, ηₜ=ηₜ, ηᵣ=ηᵣ)
            agent.state.h = habitUpdate(agent.state.h, SC[3], α)
        end
        Rwd += SC[2]
        actn = actn, μ, SC[3], SC[4]
    end

    return agent, Rwd, actn, π
end

## ask the data - model based
function agentCtrller_mixed(agent::DecisionTree{State{String,T1,T1,T1,T1}}, data::Array{Bool,1}, ϵ_cut::T; α::T=0.5, β::T=5.0, λ::T=0.0, ηₜ::T=0.5, ηᵣ::T=0.5) where {T<:Float64, T1<:Actions}
    stNm = agent.state.name
    π = softMax([agent.state.Q.A1, agent.state.Q.A2], β=β)

    if data[1]
        actn = "A1"
        agent.state.e.A1 = 1.0
        Q_old = agent.state.Q.A1
    else
        actn = "A2"
        agent.state.e.A2 = 1.0
        Q_old = agent.state.Q.A2
    end

    μ, Rwd = askEnviron(stNm, data[1:2])
    if stNm == "ξ"
        agentUpdate_mixed(agent, actn, Rwd, μ, α, λ, ηₜ, ηᵣ)
        if μ
            SC=agentCtrller_mixed(agent.μ, data[3:4], ϵ_cut, α=α, β=β, λ=λ, ηₜ=ηₜ, ηᵣ=ηᵣ)
            agentUpdate_mixed(agent, "μ", SC[3], SC[2], μ, α, λ, ηₜ, ηᵣ)
        else
            SC=agentCtrller_mixed(agent.ν, data[3:4], ϵ_cut, α=α, β=β, λ=λ, ηₜ=ηₜ, ηᵣ=ηᵣ)
            agentUpdate_mixed(agent, "ν", SC[3], SC[2], μ, α, λ, ηₜ, ηᵣ)
        end
        Rwd += SC[2]
        actn = actn, μ, SC[3], SC[4]

        ### VI(Model Based) part
        ϵ = 1.0
        while ϵ > ϵ_cut
            Q_old = agent.state.Q
            # agent update will recursively solve all trajectories
            agentUpdate_mixed(agent, "A1", α)
            agentUpdate_mixed(agent, "A2", α)
            ϵ = Δ(Q_old, agent.state.Q)
        end

    end

    return agent, Rwd, actn, π, Q_old
end

######## Dezfouli ###############

## ask the data - model free
function agentCtrller_mixed(agent::DecisionTree{State{String,T1,T1,T2,T1}}, data::Array{Bool,1}; α::T=0.5, β::T=5.0, λ::T=0.0, ηₜ::T=0.5, ηᵣ::T=0.5) where {T<:Float64, T1<:Actions, T2<:Nothing}
    stNm = agent.state.name
    π = softMax([agent.state.Q.A1, agent.state.Q.A2], β=β)

    if data[1]
        actn = "A1"
        agent.state.e.A1 = 1.0
    else
        actn = "A2"
        agent.state.e.A2 = 1.0
    end

    μ, Rwd = askEnviron(stNm, data[1:2])
    if stNm == "ξ"
        agentUpdate_mixed(agent, actn, Rwd, μ, α, λ, ηₜ, ηᵣ)
        if μ
            SC=agentCtrller_mixed(agent.μ, data[3:4], α=α, β=β, λ=λ, ηₜ=ηₜ, ηᵣ=ηᵣ)
            agentUpdate_mixed(agent, "μ", SC[3], SC[2], μ, α, λ, ηₜ, ηᵣ)
        else
            SC=agentCtrller_mixed(agent.ν, data[3:4], α=α, β=β, λ=λ, ηₜ=ηₜ, ηᵣ=ηᵣ)
            agentUpdate_mixed(agent, "ν", SC[3], SC[2], μ, α, λ, ηₜ, ηᵣ)
        end
        Rwd += SC[2]
        actn = actn, μ, SC[3], SC[4]
    end

    return agent, Rwd, actn, π
end

## ask the data - model based
function agentCtrller_mixed(agent::DecisionTree{State{String,T1,T1,T2,T1}}, data::Array{Bool,1}, ϵ_cut::T; α::T=0.5, β::T=5.0, λ::T=0.0, ηₜ::T=0.5, ηᵣ::T=0.5) where {T<:Float64, T1<:Actions, T2<:Nothing}
    stNm = agent.state.name
    π = softMax([agent.state.Q.A1, agent.state.Q.A2], β=β)

    if data[1]
        actn = "A1"
        agent.state.e.A1 = 1.0
        Q_old = agent.state.Q.A1
    else
        actn = "A2"
        agent.state.e.A2 = 1.0
        Q_old = agent.state.Q.A2
    end


    μ, Rwd = askEnviron(stNm, data[1:2])
    if stNm == "ξ"
        agentUpdate_mixed(agent, actn, Rwd, μ, α, λ, ηₜ, ηᵣ)
        if μ
            SC=agentCtrller_mixed(agent.μ, data[3:4], ϵ_cut, α=α, β=β, λ=λ, ηₜ=ηₜ, ηᵣ=ηᵣ)
            agentUpdate_mixed(agent, "μ", SC[3], SC[2], μ, α, λ, ηₜ, ηᵣ)
        else
            SC=agentCtrller_mixed(agent.ν, data[3:4], ϵ_cut, α=α, β=β, λ=λ, ηₜ=ηₜ, ηᵣ=ηᵣ)
            agentUpdate_mixed(agent, "ν", SC[3], SC[2], μ, α, λ, ηₜ, ηᵣ)
        end
        Rwd += SC[2]
        actn = actn, μ, SC[3], SC[4]

        ### VI(Model Based) part
        ϵ = 1.0
        while ϵ > ϵ_cut
            Q_old = agent.state.Q
            # agent update will recursively solve all trajectories
            agentUpdate_mixed(agent, "A1", α)
            agentUpdate_mixed(agent, "A2", α)
            ϵ = Δ(Q_old, agent.state.Q)
        end

    end

    return agent, Rwd, actn, π
end
