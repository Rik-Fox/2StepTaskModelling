# model free
function agentUpdate(agent::DecisionTree, actn::T0, Rwd::T3, μ::Union{Bool,Nothing}, α::T3) where {T0<:String, T1<:Actions, T2<:Nothing, T3<:Float64}

    s = agent.state

    # if in base node Qlearn Eq is updated by Qvalue of state landed in as R:=0 in this state
    if s.name == "ξ"
        # selecting rare and common transition Probabilities
        if actn == "A1"
            p = s.T.A1
            q = 1-(s.T.A1)
        else
            p=1-(s.T.A2)
            q = s.T.A2
        end
        # max action * T
        Q_ = p*findmax([agent.μ.state.Q.A1, agent.μ.state.Q.A2])[1] + q*findmax([agent.ν.state.Q.A1,agent.ν.state.Q.A2])[1]

        if actn == "A1"
            s.Q.A1 = (1-α)*s.Q.A1 + α*Q_
        else
            s.Q.A2 = (1-α)*s.Q.A2 + α*Q_
        end

        s = rwdUpdate(s,actn,Rwd,α)
        s.T = transitionUpdate(s.T,actn,μ,α)


    elseif s.name == "μ" || s.name == "ν"
        if actn == "A1"
            s.Q.A1 = (1-α)*s.Q.A1 + α*Rwd
        else
            s.Q.A2 = (1-α)*s.Q.A2 + α*Rwd
        end
        s = rwdUpdate(s,actn,Rwd,α)
        s.T = transitionUpdate(s.T,actn,μ,α)

    else
        throw(error("unrecognised state"))
    end

    return s
end

## Model-based update
function agentUpdate(agent::DecisionTree, actn::T0, α::Float64) where {T0<:String, T1<:Actions, T2<:Nothing}

    s= agent.state
        # if in base node Qlearn Eq is updated by Qvalue of state landed in as R:=0 in this state
        if agent.state.name == "ξ"
            # selecting rare and common transition Probabilities
            if actn == "A1"
                p = s.T.A1
                q = 1-(s.T.A1)
            else
                p=1-(s.T.A2)
                q = s.T.A2
            end
            # max action * T
            Q_ = p*findmax([agent.μ.state.Q.A1, agent.μ.state.Q.A2])[1] + q*findmax([agent.ν.state.Q.A1,agent.ν.state.Q.A2])[1]

            if actn == "A1"
                s.Q.A1 = (1-α)*s.Q.A1 + α*Q_
            else
                s.Q.A2 = (1-α)*s.Q.A2 + α*Q_
            end


            agent.ν.state = agentUpdate(agent.ν,"A1",α)
            agent.ν.state = agentUpdate(agent.ν,"A2",α)
            agent.μ.state = agentUpdate(agent.μ,"A1",α)
            agent.μ.state = agentUpdate(agent.μ,"A2",α)

        elseif agent.state.name == "μ" || agent.state.name == "ν"

            if actn == "A1"
                s.Q.A1 = (1-α)*s.Q.A1 + α*s.R.A1
            else
                s.Q.A2 = (1-α)*s.Q.A2 + α*s.R.A1
            end

        else
            throw(error("unrecognised state"))
        end

    return s
end
