# starting state
function transitionUpdate(T::Actions, actn::String, μ::Bool, α::Float64 )
    if μ
        if actn == "A1"
            Node.state.T.A1 = (1-α)*Node.state.T.A1 + α
        else
            Node.state.T.A2 = (1-α)*Node.state.T.A2
        end
    else
        if actn == "A1"
            Node.state.T.A1 = (1-α)*Node.state.T.A1
        else
            Node.state.T.A2 = (1-α)*Node.state.T.A2 + α
        end
    end

    return Node.state.T
end

# secondary states
function transitionUpdate(T::Actions, actn::String, μ::Nothing, α::Float64 )
    if actn == "A1"
        T.A1 = (1-α)*T.A1 + α
    elseif actn == "A2"
        T.A2 = (1-α)*T.A2 + α
    end

    return T
end
