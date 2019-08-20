# model free Transition Probabilities
function agentUpdate(agent::DecisionTree, actn::T0, Rwd::T3, μ::Union{Bool,Nothing}, α::T3) where {T0<:String, T1<:Actions, T2<:Nothing, T3<:Float64}

    if agent.state.name == "ξ"       # if in base node Qlearn Eq is updated by Qvalue of
                                    # state landed in as R:=0 in this state

        # selecting rare and common transition Probabilities
        if actn == "A1"
            p = agent.state.T.A1
            q = 1-(agent.state.T.A1)

            Q_ = p*findmax([agent.μ.state.Q.A1, agent.μ.state.Q.A2])[1] + q*findmax([agent.ν.state.Q.A1,agent.ν.state.Q.A2])[1] # max action * T

            agent.state.Q.A1 = (1-α)*agent.state.Q.A1 + α*Q_
        else
            p = 1-(agent.state.T.A2)
            q = agent.state.T.A2

            Q_ = p*findmax([agent.μ.state.Q.A1, agent.μ.state.Q.A2])[1] + q*findmax([agent.ν.state.Q.A1,agent.ν.state.Q.A2])[1] # max action * T

            agent.state.Q.A2 = (1-α)*agent.state.Q.A2 + α*Q_
        end

    elseif agent.state.name == "μ" || agent.state.name == "ν"

        Q_ = Rwd
        if actn == "A1"
            agent.state.Q.A1 = (1-α)*agent.state.Q.A1 + α*Q_
        else
            agent.state.Q.A2 = (1-α)*agent.state.Q.A2 + α*Q_
        end
    else
        throw(error("unrecognised state"))
    end

    agent.state.T = transitionUpdate(agent.state.T,actn,μ,α)

    return agent.state
end

## Model-based update
function agentUpdate(agent::DecisionTree, actn::T0, α::Float64) where {T0<:String, T1<:Actions, T2<:Nothing}

        # if in base node Qlearn Eq is updated by Qvalue of state landed in as R:=0 in this state
        if agent.state.name == "ξ"
            # selecting rare and common transition Probabilities
            actn == "A1" ? (p = agent.state.T.A1 ; q = 1-(agent.state.T.A1)) : (p = 1-(agent.state.T.A2) ; q = agent.state.T.A2)
            # max action * T
            Q_ = p*findmax([agent.μ.state.Q.A1, agent.μ.state.Q.A2])[1] + q*findmax([agent.ν.state.Q.A1,agent.ν.state.Q.A2])[1]
            if actn == "A1"
                agent.state.Q.A1 = (1-α)*agent.state.Q.A1 + α*Q_
            else
                agent.state.Q.A2 = (1-α)*agent.state.Q.A2 + α*Q_
            end

        elseif agent.state.name == "μ" || agent.state.name == "ν"

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

# habit update
# function agentUpdate(agent::DecisionTree{State{T0, T2, T2, T1, T1}}, actn::T0, Rwd::T3, μ::Union{Bool,Nothing}, α::T3) where {T0<:String, T1<:Actions, T2<:Nothing, T3<:Float64}
#     if actn == "A1"
#         agent.state.h.A1 = (1-α)*agent.state.h.A1 + α
#         agent.state.h.A2 = (1-α)*agent.state.h.A2
#     elseif actn == "A2"
#         agent.state.h.A1 = (1-α)*agent.state.h.A1
#         agent.state.h.A2 = (1-α)*agent.state.h.A2 + α
#     else
#         throw(ArgumentError("Action argument must be either \"A1\" or \"A2\""))
#     end
#     return agent.state
# end

# no transition probabilities
# function agentUpdate(agent::DecisionTree{State{T0, T1, T2, T2, T1}}, actn::T0, Rwd::T3, μ::Union{Bool,Nothing}, α::T3) where {T0<:String, T1<:Actions, T2<:Nothing, T3<:Float64}
#
#     agent.state.name == "ξ" ? ( μ ? Q_ = findmax([agent.μ.state.Q.A1, agent.μ.state.Q.A2])[1] : Q_ = findmax([agent.ν.state.Q.A1, agent.ν.state.Q.A2])[1] ) : Q_ = Rwd
#
#     if actn == "A1"
#         agent.state.Q.A1 = (1-α)*agent.state.Q.A1 + α*Q_
#     elseif actn == "A2"
#         agent.state.Q.A2 = (1-α)*agent.state.Q.A2 + α*Q_
#     else
#         throw(ArgumentError("action argument must be either \"A1\" or \"A2\""))
#     end
#
#     return agent.state
# end
