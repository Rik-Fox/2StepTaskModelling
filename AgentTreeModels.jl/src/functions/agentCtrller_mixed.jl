using Random

### run Epochs of Models comprised of Multiple Controllers ###

## ask the environ - model-free
function agentCtrller_mixed(agent::DecisionTree; α::T=0.5, θ::T=5.0, λ::T=0) where T<:Float64

    if typeof(agent.state.Q) == Nothing
        π = softMax([agent.state.h.A1, agent.state.h.A2],θ=θ)
    else
        π = softMax([agent.state.Q.A1, agent.state.Q.A2],θ=θ)
    end
    π > rand() ? actn = "A1" : actn = "A2"

    μ, Rwd = askEnviron(agent.state.name,actn)
    eligibilityUpdate(agent,actn,λ)
    agent.state = agentUpdate_mixed(agent,actn,Rwd,μ,α,"habit")
    if agent.state.name == "ξ"
        μ ? SC=agentCtrller_mixed(agent.μ,ctrl,α=α,θ=θ) : SC=agentCtrller_mixed(agent.ν,ctrl,α=α,θ=θ)
        Rwd += SC[2]
        actn = actn, μ, SC[3], SC[4]
    end

    return agent, Rwd, actn, π
end

## ask the environ - model based
function agentCtrller_mixed(agent::DecisionTree, ϵ_cut::T; α::T=0.5, θ::T=5.0, λ::T=0) where T<:Float64

    if typeof(agent.state.Q) == Nothing
        π = softMax([agent.state.h.A1, agent.state.h.A2],θ=θ)
    else
        π = softMax([agent.state.Q.A1, agent.state.Q.A2],θ=θ)
    end
    π > rand() ? actn = "A1" : actn = "A2"

    μ, Rwd = askEnviron(agent.state.name,actn)
    eligibilityUpdate(agent,actn,λ)
    agent.state = agentUpdate_mixed(agent,actn,Rwd,μ,α,"GD")
    if agent.state.name == "ξ"
        μ ? SC=agentCtrller_mixed(agent.μ,ctrl,α=α,θ=θ) : SC=agentCtrller_mixed(agent.ν,ctrl,α=α,θ=θ)
        Rwd += SC[2]
        actn = actn, μ, SC[3], SC[4]

        ### VI(Model Based) part
        ϵ = 1.0
        do while ϵ > ϵ_cut
            Q_old = agent.state.Q
            # agent update will recursively solve all trajectories
            agent.state = agentUpdate_mixed(agent,"A1",α)
            agent.state = agentUpdate_mixed(agent,"A2",α)
            ϵ = Δ(Q_old,agent.state.Q)
        end

    end

    return agent, Rwd, actn, π
end

## ask the data - model free
function agentCtrller_mixed(agent::DecisionTree, data::Array{Bool,1}; α::T=0.5, θ::T=5.0, λ::T=0) where T<:Float64

    if typeof(agent.state.Q) == Nothing
        π = softMax([agent.state.h.A1, agent.state.h.A2],θ=θ)
    else
        π = softMax([agent.state.Q.A1, agent.state.Q.A2],θ=θ)
    end
    data[1] ? actn = "A1" : actn = "A2"

    μ, Rwd = askEnviron(agent.state.name,data[1:2])
    eligibilityUpdate(agent,actn,λ)
    agent.state = agentUpdate_mixed(agent,actn,Rwd,μ,α,"habit")
    if agent.state.name == "ξ"
        μ ? SC=agentCtrller_mixed(agent.μ,data[3:4],ctrl,α=α,θ=θ) : SC=agentCtrller_mixed(agent.ν,data[3:4],ctrl,α=α,θ=θ)
        Rwd += SC[2]
        actn = actn, μ, SC[3], SC[4]
    end


    return agent, Rwd, actn, π
end

## ask the data - model based
function agentCtrller_mixed(agent::DecisionTree, data::Array{Bool,1}, ϵ_cut::T; α::T=0.5, θ::T=5.0, λ::T=0) where T<:Float64

    if typeof(agent.state.Q) == Nothing
        π = softMax([agent.state.h.A1, agent.state.h.A2],θ=θ)
    else
        π = softMax([agent.state.Q.A1, agent.state.Q.A2],θ=θ)
    end
    data[1] ? actn = "A1" : actn = "A2"

    μ, Rwd = askEnviron(agent.state.name,data[1:2])
    eligibilityUpdate(agent,actn,λ)
    agent.state = agentUpdate_mixed(agent,actn,Rwd,μ,α,"GD")
    if agent.state.name == "ξ"
        μ ? SC=agentCtrller_mixed(agent.μ,data[3:4],ctrl,α=α,θ=θ) : SC=agentCtrller_mixed(agent.ν,data[3:4],ctrl,α=α,θ=θ)
        Rwd += SC[2]
        actn = actn, μ, SC[3], SC[4]

        ### VI(Model Based) part
        ϵ = 1.0
        do while ϵ > ϵ_cut
            Q_old = agent.state.Q
            # agent update will recursively solve all trajectories
            agent.state = agentUpdate_mixed(agent,"A1",α)
            agent.state = agentUpdate_mixed(agent,"A2",α)
            ϵ = Δ(Q_old,agent.state.Q)
        end

    end

    return agent, Rwd, actn, π
end
