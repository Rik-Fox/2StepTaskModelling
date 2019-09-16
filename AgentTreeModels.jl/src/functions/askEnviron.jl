using Random

function askEnviron(state::String, actn::String)
    if state == "ξ"
        actn == "A1" ? (rand() > 0.7 ? μ = false : μ = true) : (rand() > 0.7 ? μ = true : μ = false)
        R = 0.0
    elseif state == "μ"
        μ = Nothing()
        actn == "A1" ? R = rwd(0.8) : R = rwd(0.0)
    elseif state == "ν"
        μ = Nothing()
        actn == "A1" ? R = rwd(0.2) : R = rwd(0.0)
    end

    return μ, R
end

function askEnviron(state::String, data::Array{Bool,1})
    if state == "ξ"
        data[1] ? (data[2] ? μ = false : μ = true) : (data[2] ? μ = true : μ = false)
        R = 0.0
    else
        μ = Nothing()
        R = Float64(data[2])
    end

    return μ, R
end
