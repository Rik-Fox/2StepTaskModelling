function habitUpdate(h::Actions, actn::String, α::Float64)
    if actn == "A1"
        h.A1 = (1 - α) * h.A1 + α
        h.A2 = (1 - α) * h.A2
    elseif actn == "A2"
        h.A1 = (1 - α) * h.A1
        h.A2 = (1 - α) * h.A2 + α
    else
        throw(ArgumentError("Action argument must be either \"A1\" or \"A2\""))
    end

    return h
end
