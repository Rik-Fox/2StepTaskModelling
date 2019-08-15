import Random

# habit
function agentCtrller(agent::DecisionTree; α::T=0.5, θ::T=5.0) where T<:AbstractFloat

    if typeof(agent.state.Q) == Nothing
        π = softMax([agent.state.h.A1, agent.state.h.A2],θ=θ)
    else
        π = softMax([agent.state.Q.A1, agent.state.Q.A2],θ=θ)
    end
    π >= rand() ? actn = "A1" : actn = "A2"

    μ, Rwd = askEnviron(agent.state.name,actn)
    agent.state.R = Rwd
    agent.state = agentUpdate(agent,actn,μ,α)
    if agent.state.name == "ξ"
        μ ? SC=agentCtrller(agent.μ,α=α,θ=θ) : SC=agentCtrller(agent.ν,α=α,θ=θ)
        Rwd = SC[2]
        actn = actn, μ, SC[3], SC[4]
    end
    return agent, Rwd, actn, π
end

function agentCtrller(agent::DecisionTree, data::Array{Bool,1}; α::T=0.5, θ::T=5.0) where T<:AbstractFloat

    if typeof(agent.state.Q) == Nothing
        π = softMax([agent.state.h.A1, agent.state.h.A2],θ=θ)
    else
        π = softMax([agent.state.Q.A1, agent.state.Q.A2],θ=θ)
    end
    data[1] ? actn = "A1" : actn = "A2"

    μ, Rwd = askEnviron(agent.state.name,data[1:2])
    agent.state.R = Rwd
    agent.state = agentUpdate(agent,actn,μ,α)
    if agent.state.name == "ξ"
        μ ? SC=agentCtrller(agent.μ,data[3:4],α=α,θ=θ) : SC=agentCtrller(agent.ν,data[3:4],α=α,θ=θ)
        Rwd = SC[2]
        actn = actn, μ, SC[3], SC[4]
    end
    return agent, Rwd, actn, π
end
