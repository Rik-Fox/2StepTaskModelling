function replacetraceUpdate(agent::DecisionTree, λ::T, δ::T) where T<:Float64

    function f(s::State, λ::T, δ::T) where T<:Float64
        ## find best action
        a_star = Bool(findmax([s.Q.A1, s.Q.A1])[2]-1)
        ## Update Q with trace
        s.Q.A1 += δ*s.e.A1
        s.Q.A2 += δ*s.e.A2
        ## decay best action and set other(s) = 0
        if a_star
            s.e.A1 *= λ
            s.e.A2 *= λ
        else
            s.e.A1 *= λ
            s.e.A2 *= λ
        end
    end

    ### for all states
    f(agent.state, λ, δ)
    f(agent.μ.state, λ, δ)
    f(agent.ν.state, λ, δ)

    return agent
end
