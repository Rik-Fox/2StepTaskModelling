function createData(agent::DecisionTree, MB::Bool; N::Int=100, α::T=0.5, θ::T=5.0) where T<:Float64
    dat = Array{Any,2}(undef,(N,4))
    for i = 1:N
        x = agentCtrller(agent,MB,α=α,θ=θ)
        dat[i,:] = [x[3][1],x[3][2],x[3][3],x[2]]
    end
    Data = [[d=="A1" for d in dat[:,1]] [d for d in dat[:,2]] [d=="A1" for d in dat[:,3]] [d==1 for d in dat[:,4]]]

    return Data
end
