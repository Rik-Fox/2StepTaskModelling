function createData_mixed(agent::DecisionTree; N::Int=100, α::T=0.5, θ::T=5.0) where T<:Float64
    dat = Array{Any,2}(undef,(N,4))
    P𝑮 = ones(N)
    r𝑮 = 0.0
    r0 = 0.0
    h_avg = 0.0

    for i=1:N
        P𝑮[i] = 1/( 1 + exp(abs(r𝑮 - r0) - abs(h_avg^2)) )

        if rand() < P𝑮[i]
            ctrl="GD"
            x = agentCtrller_mixed(agent,ctrl,α=α,θ=θ)
            agent = x[1]
            dat[i,:] = [x[3][1],x[3][2],x[3][3],x[2]]
            r𝑮 = (1-α)*r𝑮 + α*x[2]
        else
            ctrl="habit"
            x = agentCtrller_mixed(agent,ctrl,α=α,θ=θ)
            agent = x[1]
            dat[i,:] = [x[3][1],x[3][2],x[3][3],x[2]]
            h_avg = sum([agent.state.Q.A1, agent.state.Q.A2, agent.μ.state.Q.A1, agent.μ.state.Q.A2, agent.ν.state.Q.A1, agent.ν.state.Q.A2])/6
        end
        r0 = (1-α)*r0 + α*x[2]
    end
    Data = [[d=="A1" for d in dat[:,1]] [d for d in dat[:,2]] [d=="A1" for d in dat[:,3]] [d==1 for d in dat[:,4]]]

    return Data
end
