##Miller

function runSim_mixed(agent::DecisionTree{State{String, T1, T1, T1, T1}}, data::Array{Bool,2}; α::T=0.5, θ::T=5.0) where {T<:Float64, T1<:Actions, T2<:Nothing}
    n = length(data[:,1])
    # pre allocating data arrays
    epoch_Q = zeros(6,n)
    epoch_T = zeros(6,n)
    epoch_H = zeros(6,n)
    Rwd = zeros(n)
    p1 = zeros(n)
    p2 = zeros(n)
    actn1 = Array{AbstractString,1}(undef,n)
    actn2 = Array{AbstractString,1}(undef,n)

    #HWV.state.T.A1, HWV.state.T.A2, HWV.A1.state.T.A1, HWV.A1.state.T.A2, HWV.A2.state.T.A1, HWV.A2.state.T.A2 = 0.7, 0.7, 1.0, 1.0, 1.0, 1.0
    ## Param init conditions
    P𝑮 = ones(n)
    r𝑮 = 0.0
    r0 = 0.0
    h_avg = 0.0

    for i=1:n
        P𝑮[i] = 1/( 1 + exp(abs(r𝑮 - r0) - abs(h_avg^2)) )

        if rand() < P𝑮[i]
            ctrl="GD"
            itr = agentCtrller_mixed(agent,data[i,:],ctrl,α=α,θ=θ,)
            agent = itr[1]
            Rwd[i],p1[i],p2[i],actn1[i],actn2[i] = itr[2],itr[4],itr[3][4],itr[3][1],itr[3][3]
            r𝑮 = (1-α)*r𝑮 + α*Rwd[i]
        else
            ctrl="habit"
            itr = agentCtrller_mixed(agent,data[i,:],ctrl,α=α,θ=θ)
            agent = itr[1]
            Rwd[i],p1[i],p2[i],actn1[i],actn2[i] = itr[2],itr[4],itr[3][4],itr[3][1],itr[3][3]
            h_avg = sum([agent.state.h.A1, agent.state.h.A2, agent.μ.state.h.A1, agent.μ.state.h.A2, agent.ν.state.h.A1, agent.ν.state.h.A2])/6
        end
        r0 = (1-α)*r0 + α*Rwd[i]
        epoch_Q[:,i] = [agent.state.Q.A1, agent.state.Q.A2, agent.μ.state.Q.A1, agent.μ.state.Q.A2, agent.ν.state.Q.A1, agent.ν.state.Q.A2]
        epoch_T[:,i] = [agent.state.T.A1, agent.state.T.A2, agent.μ.state.T.A1, agent.μ.state.T.A2, agent.ν.state.T.A1, agent.ν.state.T.A2]
        epoch_H[:,i] = [agent.state.h.A1, agent.state.h.A2, agent.μ.state.h.A1, agent.μ.state.h.A2, agent.ν.state.h.A1, agent.ν.state.h.A2]
    end

    return agent, (Rwd, p1, p2, actn1, actn2), (epoch_Q, epoch_T, epoch_H), P𝑮
end

####### DB

function runSim_mixed(agent::DecisionTree{State{String, T1, T1, T2, T1}}, data::Array{Bool,2}; α::T=0.5, θ::T=5.0) where {T<:Float64, T1<:Actions, T2<:Nothing}
    n = length(data[:,1])
    # pre allocating data arrays
    epoch_Q = zeros(6,n)
    epoch_T = zeros(6,n)
    Rwd = zeros(n)
    p1 = zeros(n)
    p2 = zeros(n)
    actn1 = Array{AbstractString,1}(undef,n)
    actn2 = Array{AbstractString,1}(undef,n)

    #HWV.state.T.A1, HWV.state.T.A2, HWV.A1.state.T.A1, HWV.A1.state.T.A2, HWV.A2.state.T.A1, HWV.A2.state.T.A2 = 0.7, 0.7, 1.0, 1.0, 1.0, 1.0
    ## Param init conditions
    P𝑮 = ones(n)
    r𝑮 = 0.0
    r0 = 0.0
    h_avg = 0.0

    for i=1:n
        P𝑮[i] = 1/( 1 + exp(abs(r𝑮 - r0) - abs(h_avg^2)) )

        if rand() < P𝑮[i]
            ctrl="GD"
            itr = agentCtrller_mixed(agent,data[i,:],ctrl,α=α,θ=θ)
            agent = itr[1]
            Rwd[i],p1[i],p2[i],actn1[i],actn2[i] = itr[2],itr[4],itr[3][4],itr[3][1],itr[3][3]
            r𝑮 = (1-α)*r𝑮 + α*Rwd[i]
        else
            ctrl="habit"
            itr = agentCtrller_mixed(agent,data[i,:],ctrl,α=α,θ=θ)
            agent = itr[1]
            Rwd[i],p1[i],p2[i],actn1[i],actn2[i] = itr[2],itr[4],itr[3][4],itr[3][1],itr[3][3]
            h_avg = sum([agent.state.Q.A1, agent.state.Q.A2, agent.μ.state.Q.A1, agent.μ.state.Q.A2, agent.ν.state.Q.A1, agent.ν.state.Q.A2])/6
        end
        r0 = (1-α)*r0 + α*Rwd[i]
        epoch_Q[:,i] = [agent.state.Q.A1, agent.state.Q.A2, agent.μ.state.Q.A1, agent.μ.state.Q.A2, agent.ν.state.Q.A1, agent.ν.state.Q.A2]
        epoch_T[:,i] = [agent.state.T.A1, agent.state.T.A2, agent.μ.state.T.A1, agent.μ.state.T.A2, agent.ν.state.T.A1, agent.ν.state.T.A2]
    end

    return agent, (Rwd, p1, p2, actn1, actn2), (epoch_Q, epoch_T), P𝑮
end
