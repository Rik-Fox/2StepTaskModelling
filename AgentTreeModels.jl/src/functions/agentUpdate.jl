### root node update
function agentUpdate(agent::DecisionTree, actn::String, μ::Bool, α::T, λ::T) where T <: Float64

    s = agent.state
    # selecting rare and common transition Probabilities
    if actn == "A1"
        p = s.T.A1
        q = 1 - (s.T.A1)
        s.e.A1 = 1.0
    else
        p = 1 - (s.T.A2)
        q = s.T.A2
        s.e.A2 = 1.0
    end
    # δ = r(s) + γ*(T(s,a,s')*Q(s',max a')) - Q_t
    δ = (p * findmax([agent.μ.state.Q.A1, agent.μ.state.Q.A2])[1] + q * findmax([agent.ν.state.Q.A1, agent.ν.state.Q.A2])[1])

    replacetraceUpdate(agent, λ, α, δ)

    agent.state = s

    return agent
end

######### model based

### Direct update
function agentUpdate(agent::DecisionTree, actn::String, μ::Bool) where T <: Float64

    # selecting rare and common transition Probabilities
    if actn == "A1"
        agent.state.Q.A1 = agent.state.T.A1 * findmax([agent.μ.state.Q.A1, agent.μ.state.Q.A2])[1] + (1 - (agent.state.T.A1)) * findmax([agent.ν.state.Q.A1, agent.ν.state.Q.A2])[1]
    else
        agent.state.Q.A2 = (1 - (agent.state.T.A2)) * findmax([agent.μ.state.Q.A1, agent.μ.state.Q.A2])[1] + agent.state.T.A2 * findmax([agent.ν.state.Q.A1, agent.ν.state.Q.A2])[1]
    end
    # δ = r(s) + γ*(T(s,a)*Q(s,max action)) - Q_t+1

    return agent
end
