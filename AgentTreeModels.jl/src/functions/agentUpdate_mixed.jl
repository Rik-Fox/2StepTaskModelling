###### Model-Based Update
function agentUpdate_mixed(agent::DecisionTree, actn::String, α::Float64)

        s = agent.state
        # if in base node Qlearn Eq is updated by Qvalue of state landed in as R:=0 in this state
        if s.name == "ξ"

            # selecting rare and common transition Probabilities
            # T(s,a,s') ==> p for s'=μ and q for s'=ν
            actn == "A1" ? (p = s.T.A1 ; q = 1-(s.T.A1)) : (p=1-(s.T.A2) ; q = s.T.A2)

            # max action * T
            Q_ = p*findmax([agent.μ.state.Q.A1, agent.μ.state.Q.A2])[1] + q*findmax([agent.ν.state.Q.A1,agent.ν.state.Q.A2])[1]

            # apply to Q(s,a)
            if actn == "A1"
                s.Q.A1 = (1-α)*s.Q.A1 + α*Q_
            else
                s.Q.A2 = (1-α)*s.Q.A2 + α*Q_
            end

            agentUpdate_mixed(agent.ν,"A1",α)
            agentUpdate_mixed(agent.ν,"A2",α)
            agentUpdate_mixed(agent.μ,"A1",α)
            agentUpdate_mixed(agent.μ,"A2",α)

        elseif s.name == "μ" || s.name == "ν"

            # apply predicted R to Q(s,a)
            if actn == "A1"
                s.Q.A1 = (1-α)*s.Q.A1 + α*s.R.A1
            else
                s.Q.A2 = (1-α)*s.Q.A2 + α*s.R.A2
            end
        else
            throw(error("unrecognised state"))
        end
        agent.state = s
    return agent
end


####### Environ/Data-Based Update
### second action update
function agentUpdate_mixed(agent::DecisionTree, stNM::T0, actn::T0, Rwd::T, μ::Union{Bool,Nothing}, α::T, λ::T, ηₜ::T, ηᵣ::T) where {T0<:String, T<:Float64, T1<:Actions, T2<:Nothing}



    # if in base node Qlearn Eq is updated by Qvalue of state landed in as R:=0 in this state
    if stNM == "μ"
        s = agent.μ.state
        actn == "A1" ? s.e.A1 = 1.0 : s.e.A2 = 1.0
        replacetraceUpdate(agent, λ, α*Rwd)
        s = rwdUpdate(s, actn, Rwd, ηᵣ)
        s.T = transitionUpdate(s.T, actn, μ, ηₜ)
    elseif stNM == "ν"
        s = agent.ν.state
        actn == "A1" ? s.e.A1 = 1.0 : s.e.A2 = 1.0
        replacetraceUpdate(agent, λ, α*Rwd)
        s = rwdUpdate(s, actn, Rwd, ηᵣ)
        s.T = transitionUpdate(s.T, actn, μ, ηₜ)
    end

    agent.state = s
    return agent
end

### root action update
function agentUpdate_mixed(agent::DecisionTree, actn::T0, Rwd::T, μ::Union{Bool,Nothing}, α::T, λ::T, ηₜ::T, ηᵣ::T) where {T0<:String, T<:Float64, T1<:Actions, T2<:Nothing}

    # if in base node Qlearn Eq is updated by Qvalue of state landed in as R:=0 in this state
        s = agent.state
        # selecting rare and common transition Probabilities
        if actn == "A1"
            p = s.T.A1
            q = 1-(s.T.A1)
            s.e.A1 = 1.0
            Qt = s.Q.A2
        else
            p=1-(s.T.A2)
            q = s.T.A2
            s.e.A2 = 1.0
            Qt = s.Q.A2
        end
        # max action * T
        δ = α*((p*findmax([agent.μ.state.Q.A1, agent.μ.state.Q.A2])[1] + q*findmax([agent.ν.state.Q.A1,agent.ν.state.Q.A2])[1])-Qt)

        replacetraceUpdate(agent, λ, δ)
        s = rwdUpdate(s, actn, Rwd, ηᵣ)
        s.T = transitionUpdate(s.T, actn, μ, ηₜ)

        agent.state = s
    return agent
end
