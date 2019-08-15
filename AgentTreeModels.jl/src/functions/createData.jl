function createData(agent::DecisionTree=buildAgent(2); N::Int=100, α::Float64=0.5, θ::Float64=5.0)
    dat = Array{Any,2}(undef,(N,4))
    for i = 1:N
        x = agentCtrller(agent,α=α,θ=θ)
        dat[i,:] = [x[3][1],x[3][2],x[3][3],x[2]]
    end
    Data = [[d=="A1" for d in dat[:,1]] [d for d in dat[:,2]] [d=="A1" for d in dat[:,3]] [d==1 for d in dat[:,4]]]

    return Data
end
