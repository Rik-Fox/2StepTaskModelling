################### Agent Value Updates ############################################################
function habitUpdate(h::Actions,actn::String,α::Float64)

    if actn == "A1"
        h.A1 = (1-α)*h.A1 + α
        h.A2 = (1-α)*h.A2
    elseif actn == "A2"
        h.A1 = (1-α)*h.A1
        h.A2 = (1-α)*h.A2 + α
    else
        throw(ArgumentError("Action argument must be either \"A1\" or \"A2\""))
    end
    return h
end

function QUpdate(Node::DecisionTree, actn::String, μ::Union{Bool,Nothing}, α::Float64)

    if Node.state.name == "ξ"
        μ ? Q_ = findmax([Node.μ.state.Q.A1, Node.μ.state.Q.A2])[1] : Q_ = findmax([Node.ν.state.Q.A1, Node.ν.state.Q.A2])[1]
    elseif Node.state.name == "μ" || Node.state.name == "ν"
        Q_ = Node.state.R
    else
        throw(error("unrecognised state"))
    end

    if actn == "A1"
        Node.state.Q.A1 = (1-α)*Node.state.Q.A1 + α*Q_
    elseif actn == "A2"
        Node.state.Q.A2 = (1-α)*Node.state.Q.A2 + α*Q_
    else
        throw(ArgumentError("action argument must be either \"A1\" or \"A2\""))
    end

    return Node.state.Q
end

function modelledQUpdate(Node::DecisionTree, actn::String, μ::Union{Bool,Nothing}, α::Float64)

    if Node.state.name == "ξ"       # if in base node Qlearn Eq is updated by Qvalue of
                                    # state landed in as R:=0 in this state
        actn == "A1" ? (p = Node.state.T.A1 ; q = 1-(Node.state.T.A1)) : (p = 1-(Node.state.T.A2) ; q = Node.state.T.A2)           # selecting rare and common transition Probabilities
                Q_ = p*findmax([Node.μ.state.Q.A1, Node.μ.state.Q.A2])[1] + q*findmax([Node.ν.state.Q.A1,Node.ν.state.Q.A2])[1] # max action * T
        if μ #actn == "A1" #μ
            Node.state.Q.A1 = (1-α)*Node.state.Q.A1 + α*Q_
        else
            Node.state.Q.A2 = (1-α)*Node.state.Q.A2 + α*Q_
        end
    elseif Node.state.name == "μ" || Node.state.name == "ν"

        Q_ = Node.state.R
        if actn == "A1"
            Node.state.Q.A1 = (1-α)*Node.state.Q.A1 + α*Q_
        else
            Node.state.Q.A2 = (1-α)*Node.state.Q.A2 + α*Q_
        end
    else
        throw(error("unrecognised state"))
    end

    return Node.state.Q
end

function transitionUpdate(Node::DecisionTree,actn::String,μ::Union{Bool,Nothing},α::Float64)
    if Node.state.name == "ξ"
        μ ? (actn == "A1" ? Node.state.T.A1 = (1-α)*Node.state.T.A1 + α : Node.state.T.A2 = (1-α)*Node.state.T.A2) : (actn == "A1" ? Node.state.T.A1 = (1-α)*Node.state.T.A1 : Node.state.T.A2 = (1-α)*Node.state.T.A2 + α)
    elseif Node.state.name == "μ" || Node.state.name == "ν"
        if actn == "A1"
            Node.state.T.A1 = (1-α)*Node.state.T.A1 + α
        elseif actn == "A2"
            Node.state.T.A2 = (1-α)*Node.state.T.A2 + α
        end
    else
        throw(error("unrecognised state"))
    end

    return Node.state.T
end
