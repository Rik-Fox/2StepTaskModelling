function replacetraceUpdate(agent::DecisionTree, λ::T, α::T, δ::T) where T<:Float64
    # if agent.state.e.A1 >= λ^2
        agent.state.Q.A1 = (1-α*agent.state.e.A1)*agent.state.Q.A1 + α*δ*agent.state.e.A1
    # end
    # if agent.state.e.A2 >= λ^2
        agent.state.Q.A2 = (1-α*agent.state.e.A2)*agent.state.Q.A2 + α*δ*agent.state.e.A2
    # end
    #
    # if agent.μ.state.e.A1 >= λ^2
        agent.μ.state.Q.A1 = (1-α*agent.μ.state.e.A1)*agent.μ.state.Q.A1 + α*δ*agent.μ.state.e.A1
    # end
    # if agent.μ.state.e.A2 >= λ^2
        agent.μ.state.Q.A2 = (1-α*agent.μ.state.e.A2)*agent.μ.state.Q.A2 + α*δ*agent.μ.state.e.A2
    # end
    #
    # if agent.ν.state.e.A1 >= λ^2
        agent.ν.state.Q.A1 = (1-α*agent.ν.state.e.A1)*agent.ν.state.Q.A1 + α*δ*agent.ν.state.e.A1
    # end
    # if agent.ν.state.e.A2 >= λ^2
        agent.ν.state.Q.A2 = (1-α*agent.ν.state.e.A2)*agent.ν.state.Q.A2 + α*δ*agent.ν.state.e.A2
    # end
    #####################

    agent.state.e.A1 *= λ
    agent.state.e.A2 *= λ

    agent.μ.state.e.A1 *= λ
    agent.μ.state.e.A2 *= λ

    agent.ν.state.e.A1 *= λ
    agent.ν.state.e.A2 *= λ

    return agent
end
