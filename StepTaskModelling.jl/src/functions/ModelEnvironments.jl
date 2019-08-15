using Random

################### Environment ####################################################################
function taskEval(state::String,actn::String)
    if state == "ξ"
        actn == "A1" ? (rand() < 0.7 ? μ = true : μ = false) : (rand() < 0.7 ? μ = false : μ = true)
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

function taskRead(state::String,data::Array{Bool,1})
    if state == "ξ"
        data[1] ? (data[2] ? μ = false : μ = true) : (data[2] ? μ = true : μ = false)
        R = 0.0
    else
        μ = Nothing()
        R = Int(data[2])
    end

    return μ, R
end

function taskCreateData(f::Function;agent::DecisionTree=buildAgent(2),N::Int=1000,α::Float64=0.5,θ::Float64=5.0)
    dat = Array{Any,2}(undef,(N,4))
    for i = 1:N
        x = f(agent,α=α,θ=θ)
        dat[i,:] = [x[3][1],x[3][2],x[3][3],x[2]]
    end
    Data = [[d=="A1" for d in dat[:,1]] [d for d in dat[:,2]] [d=="A1" for d in dat[:,3]] [d==1 for d in dat[:,4]]]

    return Data
end

# test = taskCreateData(agent)
#
# test[:,4]
