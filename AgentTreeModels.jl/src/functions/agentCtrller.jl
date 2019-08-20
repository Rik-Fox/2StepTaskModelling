import Random

## model based

function agentCtrller(agent::DecisionTree, MB::Bool; α::T=0.5, θ::T=5.0, M::Int64=100) where T<:Float64

    if typeof(agent.state.Q) == Nothing
        π = softMax([agent.state.h.A1, agent.state.h.A2],θ=θ)
    else
        π = softMax([agent.state.Q.A1, agent.state.Q.A2],θ=θ)
    end
    π >= rand() ? actn = "A1" : actn = "A2"

    μ, Rwd = askEnviron(agent.state.name,actn)
    agent.state = agentUpdate(agent,actn,Rwd,μ,α)

    if agent.state.name == "ξ"
        μ ? SC=agentCtrller(agent.μ,MB,α=α,θ=θ) : SC=agentCtrller(agent.ν,MB,α=α,θ=θ)
        Rwd += SC[2]
        actn = actn, μ, SC[3], SC[4]

        if MB
            for i=1:M   ## Random s and random a for tree-search/DYNA-Q
                s_rv = rand()
                if s_rv < 0.34
                    rand() > 0.5 ? a = "A1" : a = "A2"
                    agent.state = agentUpdate(agent,a,α)
                elseif s_rv < 0.67
                    rand() > 0.5 ? a = "A1" : a = "A2"
                    agent.μ.state = agentUpdate(agent.μ,a,α)
                else
                    rand() > 0.5 ? a = "A1" : a = "A2"
                    agent.ν.state = agentUpdate(agent.ν,a,α)
                end
            end
        end
    end
    return agent, Rwd, actn, π
end

function agentCtrller(agent::DecisionTree, data::Array{Bool,1}, MB::Bool; α::T=0.5, θ::T=5.0) where T<:AbstractFloat

    if typeof(agent.state.Q) == Nothing
        π = softMax([agent.state.h.A1, agent.state.h.A2],θ=θ)
    else
        π = softMax([agent.state.Q.A1, agent.state.Q.A2],θ=θ)
    end
    data[1] ? actn = "A1" : actn = "A2"

    μ, Rwd = askEnviron(agent.state.name,data[1:2])
    agent.state = agentUpdate(agent,actn,Rwd,μ,α)
    if agent.state.name == "ξ"
        μ ? SC=agentCtrller(agent.μ,data[3:4],MB,α=α,θ=θ) : SC=agentCtrller(agent.ν,data[3:4],MB,α=α,θ=θ)
        Rwd += SC[2]
        actn = actn, μ, SC[3], SC[4]

        if MB
            for i=1:M  ## Random s and random a for tree-search/DYNA-Q
                s_rv = rand()
                if s_rv < 0.34
                    rand() > 0.5 ? a = "A1" : a = "A2"
                    agent.state = agentUpdate(agent,a,α)
                elseif s_rv < 0.67
                    rand() > 0.5 ? a = "A1" : a = "A2"
                    agent.μ.state = agentUpdate(agent.μ,a,α)
                else
                    rand() > 0.5 ? a = "A1" : a = "A2"
                    agent.ν.state = agentUpdate(agent.ν,a,α)
                end
            end
        end
    end

    return agent, Rwd, actn, π
end
