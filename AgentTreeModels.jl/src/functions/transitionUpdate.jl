# starting state
function transitionUpdate(T::Actions, actn::String, μ::Bool, α::Float64 )
    if μ
        actn == "A1" ? T.A1 = (1-α)*T.A1 + α : T.A2 = (1-α)*T.A2
    else
        actn == "A1" ? T.A1 = (1-α)*T.A1 : T.A2 = (1-α)*T.A2 + α
    end

    return T
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
