# habit update
function agentUpdate(agent::DecisionTree{State{T0, T2, T2, T1, T}}, actn::T0, μ, α::T) where {T0<:AbstractString, T<:AbstractFloat, T1<:Actions, T2<:Nothing}
    if actn == "A1"
        agent.state.h.A1 = (1-α)*agent.state.h.A1 + α
        agent.state.h.A2 = (1-α)*agent.state.h.A2
    elseif actn == "A2"
        agent.state.h.A1 = (1-α)*agent.state.h.A1
        agent.state.h.A2 = (1-α)*agent.state.h.A2 + α
    else
        throw(ArgumentError("Action argument must be either \"A1\" or \"A2\""))
    end
    return agent.state
end

# cached values
function agentUpdate(agent::DecisionTree{State{T0, T1, T2, T2, T}}, actn::T0, μ, α::T) where {T0<:AbstractString, T<:AbstractFloat, T1<:Actions, T2<:Nothing}

    agent.state.name == "ξ" ? ( μ ? Q_ = findmax([agent.μ.state.Q.A1, agent.μ.state.Q.A2])[1] : Q_ = findmax([agent.ν.state.Q.A1, agent.ν.state.Q.A2])[1] ) : Q_ = agent.state.R

    if actn == "A1"
        agent.state.Q.A1 = (1-α)*agent.state.Q.A1 + α*Q_
    elseif actn == "A2"
        agent.state.Q.A2 = (1-α)*agent.state.Q.A2 + α*Q_
    else
        throw(ArgumentError("action argument must be either \"A1\" or \"A2\""))
    end

    return agent.state
end

# Transition Probabilities
function agentUpdate(agent::DecisionTree{State{T0, T1, T1, T2, T}}, actn::T0, μ, α::T; transUpdate::Bool=false) where {T0<:AbstractString, T<:AbstractFloat, T1<:Actions, T2<:Nothing}

    if agent.state.name == "ξ"       # if in base node Qlearn Eq is updated by Qvalue of
                                    # state landed in as R:=0 in this state
        actn == "A1" ? (p = agent.state.T.A1 ; q = 1-(agent.state.T.A1)) : (p = 1-(agent.state.T.A2) ; q = agent.state.T.A2)           # selecting rare and common transition Probabilities
                Q_ = p*findmax([agent.μ.state.Q.A1, agent.μ.state.Q.A2])[1] + q*findmax([agent.ν.state.Q.A1,agent.ν.state.Q.A2])[1] # max action * T
        if μ
            agent.state.Q.A1 = (1-α)*agent.state.Q.A1 + α*Q_
        else
            agent.state.Q.A2 = (1-α)*agent.state.Q.A2 + α*Q_
        end
    elseif agent.state.name == "μ" || agent.state.name == "ν"

        Q_ = agent.state.R
        if actn == "A1"
            agent.state.Q.A1 = (1-α)*agent.state.Q.A1 + α*Q_
        else
            agent.state.Q.A2 = (1-α)*agent.state.Q.A2 + α*Q_
        end
    else
        throw(error("unrecognised state"))
    end

    if transUpdate
        agent.state.T = transitionUpdate(agent.state.T,actn,μ,α)
    end

    return agent.state
end
