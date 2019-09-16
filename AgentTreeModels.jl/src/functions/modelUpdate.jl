#### DYNA Tree Search method
function modelUpdate(agent::DecisionTree{State{String,Actions,Nothing}}, α::Float64, μ::Bool, M::Int64)
    for i = 1:M   ## Random s and random a for tree-search/DYNA-Q
        s_rv = rand()
        if s_rv < 0.34
            rand() > 0.5 ? a = "A1" : a = "A2"
            agentUpdate(agent, a, μ, α)
        elseif s_rv < 0.67
            rand() > 0.5 ? a = "A1" : a = "A2"
            agent.μ = agentUpdate(agent.μ, a, α)
        else
            rand() > 0.5 ? a = "A1" : a = "A2"
            agent.ν = agentUpdate(agent.ν, a, α)
        end
    end
    return agent
end



### Value iteration - converge on policy
function modelUpdate(agent::DecisionTree, α::T, μ::Bool, ϵ_cut::T) where T <: Float64
    ϵ = 1.0
    while ϵ > ϵ_cut
        Q_old = sum([agent.state.Q.A1, agent.state.Q.A2, agent.μ.state.Q.A1, agent.μ.state.Q.A2, agent.ν.state.Q.A1, agent.ν.state.Q.A2])
        # agent update will recursively solve all trajectories
        agentUpdate(agent, "A1", μ, α)
        agentUpdate(agent, "A2", μ, α)
        agent.μ = agentUpdate(agent.μ, "A1", α)
        agent.μ = agentUpdate(agent.μ, "A2", α)
        agent.ν = agentUpdate(agent.ν, "A1", α)
        agent.ν = agentUpdate(agent.ν, "A2", α)
        ϵ = abs(Q_old - sum([agent.state.Q.A1, agent.state.Q.A2, agent.μ.state.Q.A1, agent.μ.state.Q.A2, agent.ν.state.Q.A1, agent.ν.state.Q.A2]))
    end
    return agent
end
