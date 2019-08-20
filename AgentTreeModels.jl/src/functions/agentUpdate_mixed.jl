###### Model-Based Update
function agentUpdate_mixed(agent::DecisionTree, actn::String, α::Float64)


        # if in base node Qlearn Eq is updated by Qvalue of state landed in as R:=0 in this state
        if agent.state.name == "ξ"

            # selecting rare and common transition Probabilities
            if actn == "A1"
                p = agent.state.T.A1
                q = 1-(agent.state.T.A1)
            else
                p = 1-(agent.state.T.A2)
                q = agent.state.T.A2
            end

            # max action * T
            Q_ = p*findmax([agent.μ.state.Q.A1, agent.μ.state.Q.A2])[1] + q*findmax([agent.ν.state.Q.A1,agent.ν.state.Q.A2])[1]

            # apply to Q(s,a)
            if actn == "A1"
                agent.state.Q.A1 = (1-α)*agent.state.Q.A1 + α*Q_
            else
                agent.state.Q.A2 = (1-α)*agent.state.Q.A2 + α*Q_
            end

            agent.ν.state = agentUpdate_mixed(agent.ν,"A1",α)
            agent.ν.state = agentUpdate_mixed(agent.ν,"A2",α)
            agent.μ.state = agentUpdate_mixed(agent.μ,"A1",α)
            agent.μ.state = agentUpdate_mixed(agent.μ,"A2",α)

        elseif agent.state.name == "μ" || agent.state.name == "ν"

            # apply predicted R to Q(s,a)
            if actn == "A1"
                agent.state.Q.A1 = (1-α)*agent.state.Q.A1 + α*agent.state.R.A1
            else
                agent.state.Q.A2 = (1-α)*agent.state.Q.A2 + α*agent.state.R.A2
            end
        else
            throw(error("unrecognised state"))
        end

    return agent.state
end


###### Miller

function agentUpdate_mixed(agent::DecisionTree{State{T0, T1, T1, T1, T1}}, actn::T0, Rwd::T, μ::Union{Bool,Nothing}, α::T, ctrl::T0) where {T0<:String, T<:Float64, T1<:Actions}

    if ctrl == "GD"
        # if in base node Qlearn Eq is updated by Qvalue of state landed in as R:=0 in this state
        if agent.state.name == "ξ"
            # selecting rare and common transition Probabilities
            actn == "A1" ? (p = agent.state.T.A1 ; q = 1-(agent.state.T.A1)) : (p=1-(agent.state.T.A2) ; q = agent.state.T.A2)
            # max action * T
            Q_ = p*findmax([agent.μ.state.Q.A1, agent.μ.state.Q.A2])[1] + q*findmax([agent.ν.state.Q.A1,agent.ν.state.Q.A2])[1]

            if actn == "A1"
                agent.state.Q.A1 = (1-α)*agent.state.Q.A1 + α*Q_
                agent.state.R.A1 = (1-α)*agent.state.R.A1 + α*Rwd
            else
                agent.state.Q.A2 = (1-α)*agent.state.Q.A2 + α*Q_
                agent.state.R.A2 = (1-α)*agent.state.R.A2 + α*Rwd
            end

            agent.state.T = transitionUpdate(agent.state.T,actn,μ,α)


        elseif agent.state.name == "μ" || agent.state.name == "ν"

            if actn == "A1"
                agent.state.Q.A1 = (1-α)*agent.state.Q.A1 + α*Rwd
                agent.state.R.A1 = (1-α)*agent.state.R.A1 + α*Rwd
            else
                agent.state.Q.A2 = (1-α)*agent.state.Q.A2 + α*Rwd
                agent.state.R.A1 = (1-α)*agent.state.R.A1 + α*Rwd
            end

            agent.state.T = transitionUpdate(agent.state.T,actn,μ,α)

        else
            throw(error("unrecognised state"))
        end

    elseif ctrl == "habit"
        if actn == "A1"
            agent.state.h.A1 = (1-α)*agent.state.h.A1 + α
            agent.state.h.A2 = (1-α)*agent.state.h.A2
        elseif actn == "A2"
            agent.state.h.A1 = (1-α)*agent.state.h.A1
            agent.state.h.A2 = (1-α)*agent.state.h.A2 + α
        else
            throw(ArgumentError("Action argument must be either \"A1\" or \"A2\""))
        end
    else
        throw(error("Must select controller-GD/habit"))
    end

    return agent.state
end


####### DB

function agentUpdate_mixed(agent::DecisionTree{State{T0, T1, T1, T2, T1}}, actn::T0, Rwd::T, μ::Union{Bool,Nothing}, α::T, ctrl::T0) where {T0<:String, T<:Float64, T1<:Actions, T2<:Nothing}

    # if in base node Qlearn Eq is updated by Qvalue of state landed in as R:=0 in this state
        if agent.state.name == "ξ"
            # selecting rare and common transition Probabilities
            actn == "A1" ? (p = agent.state.T.A1 ; q = 1-(agent.state.T.A1)) : (p=1-(agent.state.T.A2) ; q = agent.state.T.A2)
            # max action * T
            Q_ = p*findmax([agent.μ.state.Q.A1, agent.μ.state.Q.A2])[1] + q*findmax([agent.ν.state.Q.A1,agent.ν.state.Q.A2])[1]

            if actn == "A1"
                agent.state.Q.A1 = (1-α)*agent.state.Q.A1 + α*Q_
                agent.state.R.A1 = (1-α)*agent.state.R.A1 + α*Rwd
                agent.state.e.A1 = 1
            else
                agent.state.Q.A2 = (1-α)*agent.state.Q.A2 + α*Q_
                agent.state.R.A2 = (1-α)*agent.state.R.A2 + α*Rwd
                agent.state.e.A2 = 1
            end

            agent.state.T = transitionUpdate(agent.state.T,actn,μ,α)


            if
                agent.state.e = eligibilityUpdate(agent.state.e)


        elseif agent.state.name == "μ" || agent.state.name == "ν"

            if actn == "A1"
                agent.state.Q.A1 = (1-α)*agent.state.Q.A1 + α*Rwd
                agent.state.R.A1 = (1-α)*agent.state.R.A1 + α*Rwd
            else
                agent.state.Q.A2 = (1-α)*agent.state.Q.A2 + α*Rwd
                agent.state.R.A1 = (1-α)*agent.state.R.A1 + α*Rwd
            end

            agent.state.T = transitionUpdate(agent.state.T,actn,μ,α)

        else
            throw(error("unrecognised state"))
        end

    return agent.state
end
