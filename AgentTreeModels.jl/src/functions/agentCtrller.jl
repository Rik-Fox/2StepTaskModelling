import Random

############# ASK THE ENVIRON ###############

## model free

function agentCtrller(agent::DecisionTree; α::T=0.5, β::T=5.0) where T<:Float64

    π = softMax([agent.state.Q.A1, agent.state.Q.A2],β=β)
    π >= rand() ? actn = "A1" : actn = "A2"

    μ, Rwd = askEnviron(agent.state.name,actn)
    agent.state = agentUpdate(agent,actn,Rwd,μ,α)

    if agent.state.name == "ξ"
        μ ? SC=agentCtrller(agent.μ,α=α,β=β) : SC=agentCtrller(agent.ν,α=α,β=β)
        Rwd += SC[2]
        actn = actn, μ, SC[3], SC[4]
    end

    return agent, Rwd, actn, π
end

## model based

function agentCtrller(agent::DecisionTree, M::Int64; α::T=0.5, β::T=5.0) where T<:Float64

    π = softMax([agent.state.Q.A1, agent.state.Q.A2],β=β)
    π >= rand() ? actn = "A1" : actn = "A2"

    μ, Rwd = askEnviron(agent.state.name,actn)
    agent.state = agentUpdate(agent,actn,Rwd,μ,α)

    if agent.state.name == "ξ"
        μ ? SC=agentCtrller(agent.μ,α=α,β=β) : SC=agentCtrller(agent.ν,α=α,β=β)
        Rwd += SC[2]
        actn = actn, μ, SC[3], SC[4]

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

    return agent, Rwd, actn, π
end

####### ASK THE DATA ###############

## model based

function agentCtrller(agent::DecisionTree, data::Array{Bool,1}, M::Int64; α::T=0.5, β::T=5.0) where T<:Float64

    π = softMax([agent.state.Q.A1, agent.state.Q.A2],β=β)
    data[1] ? actn = "A1" : actn = "A2"

    μ, Rwd = askEnviron(agent.state.name,actn)
    agent.state = agentUpdate(agent,actn,Rwd,μ,α)

    if agent.state.name == "ξ"
        μ ? SC=agentCtrller(agent.μ,data[3:4],α=α,β=β) : SC=agentCtrller(agent.ν,data[3:4],α=α,β=β)
        Rwd += SC[2]
        actn = actn, μ, SC[3], SC[4]

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

    return agent, Rwd, actn, π
end

## model free

function agentCtrller(agent::DecisionTree, data::Array{Bool,1}; α::T=0.5, β::T=5.0) where T<:Float64

    π = softMax([agent.state.Q.A1, agent.state.Q.A2], β=β)

    data[1] ? actn = "A1" : actn = "A2"

    μ, Rwd = askEnviron(agent.state.name,data[1:2])
    agent.state = agentUpdate(agent,actn,Rwd,μ,α)

    if agent.state.name == "ξ"
        μ ? SC=agentCtrller(agent.μ,data[3:4],α=α,β=β) : SC=agentCtrller(agent.ν,data[3:4],α=α,β=β)
        Rwd += SC[2]
        actn = actn, μ, SC[3], SC[4]
    end

    return agent, Rwd, actn, π
end
