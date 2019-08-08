############## Functions that run a full Epoch of a Raw Controller ###############################
function runHabit(; agent::DecisionTree=buildAgent(2,hm=true), data::Union{AbstractArray,Nothing}=Nothing(), n::Int=1000, α::Float64=0.5, θ::Float64=5.0 )
    if typeof(data) != Nothing
        n = length(data[:,1])
    end
    epoch_H = zeros(6,n)
    epoch_Q = zeros(6,n)
    Rwd = zeros(n)
    p1 = zeros(n)
    p2 = zeros(n)
    for i=1:n
        d = Nothing()
        if typeof(data) != Nothing
            d=data[i,:]
        end
        itr = habitCtrl(agent,data=d,α=α,θ=θ)
        Rwd[i],p1[i],p2[i] = itr[2],itr[4],itr[3][4]
        epoch_H[:,i] = [agent.state.h.A1, agent.state.h.A2, agent.μ.state.h.A1, agent.μ.state.h.A2, agent.ν.state.h.A1, agent.ν.state.h.A2]
        epoch_Q[:,i] = [agent.state.Q.A1, agent.state.Q.A2, agent.μ.state.Q.A1, agent.μ.state.Q.A2, agent.ν.state.Q.A1, agent.ν.state.Q.A2]
    end

    return agent, (Rwd, p1, p2), epoch_Q,Nothing(), epoch_H
end

function runMF(; agent::DecisionTree=buildAgent(2), data::Union{AbstractArray,Nothing}=Nothing(), n::Int=1000, α::Float64=0.5, θ::Float64=5.0)
    if typeof(data) != Nothing
        n = length(data[:,1])
    end
    epoch_Q = zeros(6,n)
    Rwd = zeros(n)
    p1 = zeros(n)
    p2 = zeros(n)
    for i=1:n
        d = Nothing()
        if typeof(data) != Nothing
            d=data[i,:]
        end
        itr = MFCtrl(agent,data=d,α=α,θ=θ)
        Rwd[i],p1[i],p2[i] = itr[2],itr[4],itr[3][4]
        epoch_Q[:,i] = [agent.state.Q.A1, agent.state.Q.A2, agent.μ.state.Q.A1, agent.μ.state.Q.A2, agent.ν.state.Q.A1, agent.ν.state.Q.A2]
    end

    return agent, (Rwd, p1, p2), epoch_Q, Nothing(), Nothing()
end

function runMB(; agent::DecisionTree=buildAgent(2,TM=true), data::Union{AbstractArray,Nothing}=Nothing(), n::Int=1000, α::Float64=0.5, θ::Float64=5.0, TM::AbstractArray = [0.7 0.7 1.0 1.0 1.0 1.0] )
    agent.state.T.A1, agent.state.T.A2, agent.μ.state.T.A1, agent.μ.state.T.A2, agent.ν.state.T.A1, agent.ν.state.T.A2 = TM[1], TM[2], TM[3], TM[4], TM[5], TM[6]
    if typeof(data) != Nothing
        n = length(data[:,1])
    end
    epoch_Q = zeros(6,n)
    epoch_T = zeros(6,n)
    Rwd = zeros(n)
    p1 = zeros(n)
    p2 = zeros(n)
    for i=1:n
        d = Nothing()
        if typeof(data) != Nothing
            d=data[i,:]
        end
        itr = MBCtrl(agent,data=d,α=α,θ=θ)
        Rwd[i],p1[i],p2[i] = itr[2],itr[4],itr[3][4]
        epoch_Q[:,i] = [agent.state.Q.A1, agent.state.Q.A2, agent.μ.state.Q.A1, agent.μ.state.Q.A2, agent.ν.state.Q.A1, agent.ν.state.Q.A2]
        epoch_T[:,i] = [agent.state.T.A1, agent.state.T.A2, agent.μ.state.T.A1, agent.μ.state.T.A2, agent.ν.state.T.A1, agent.ν.state.T.A2]
    end

    return agent, (Rwd, p1, p2), epoch_Q, epoch_T, Nothing()
end

function runGD(; agent::DecisionTree=buildAgent(2,TM=true), data::Union{AbstractArray,Nothing}=Nothing(), n::Int=1000, α::Float64=0.5, θ::Float64=5.0 )
    if typeof(data) != Nothing
        n = length(data[:,1])
    end
    epoch_Q = zeros(6,n)
    epoch_T = zeros(6,n)
    Rwd = zeros(n)
    p1 = zeros(n)
    p2 = zeros(n)
    for i=1:n
        d = Nothing()
        if typeof(data) != Nothing
            d=data[i,:]
        end
        itr = GDCtrl(agent,data=d,α=α,θ=θ)
        Rwd[i],p1[i],p2[i] = itr[2],itr[4],itr[3][4]
        epoch_Q[:,i] = [agent.state.Q.A1, agent.state.Q.A2, agent.μ.state.Q.A1, agent.μ.state.Q.A2, agent.ν.state.Q.A1, agent.ν.state.Q.A2]
        epoch_T[:,i] = [agent.state.T.A1, agent.state.T.A2, agent.μ.state.T.A1, agent.μ.state.T.A2, agent.ν.state.T.A1, agent.ν.state.T.A2]
    end

    return agent, (Rwd, p1, p2), epoch_Q, epoch_T, Nothing()
end

########### Functions that run Epochs of Models comprised of Multiple Controllers ##################

function runHWV(; agent::DecisionTree=buildAgent(2,TM=true,hm=true), data::Union{AbstractArray,Nothing}=Nothing(), n::Int=1000, α::Float64=0.5, θ::Float64=5.0 )
    if typeof(data) != Nothing
        n = length(data[:,1])
    end
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
        d = Nothing()
        if typeof(data) != Nothing
            d=data[i,:]
        end

        if rand() < P𝑮[i]
            itr = GDCtrl(agent,data=d,α=α,θ=θ)
            Rwd[i],p1[i],p2[i],actn1[i],actn2[i] = itr[2],itr[4],itr[3][4],itr[3][1],itr[3][3]
            r𝑮 = (1-α)*r𝑮 + α*Rwd[i]
        else
            itr = habitCtrl(agent,data=d,α=α,θ=θ)
            Rwd[i],p1[i],p2[i],actn1[i],actn2[i] = itr[2],itr[4],itr[3][4],itr[3][1],itr[3][3]
            h_avg = sum([agent.state.h.A1, agent.state.h.A2, agent.μ.state.h.A1, agent.μ.state.h.A2, agent.ν.state.h.A1, agent.ν.state.h.A2])/6
        end
        r0 = (1-α)*r0 + α*Rwd[i]
        epoch_Q[:,i] = [agent.state.Q.A1, agent.state.Q.A2, agent.μ.state.Q.A1, agent.μ.state.Q.A2, agent.ν.state.Q.A1, agent.ν.state.Q.A2]
        epoch_T[:,i] = [agent.state.T.A1, agent.state.T.A2, agent.μ.state.T.A1, agent.μ.state.T.A2, agent.ν.state.T.A1, agent.ν.state.T.A2]
        epoch_H[:,i] = [agent.state.h.A1, agent.state.h.A2, agent.μ.state.h.A1, agent.μ.state.h.A2, agent.ν.state.h.A1, agent.ν.state.h.A2]
    end

    return agent, (Rwd, p1, p2, actn1, actn2), epoch_Q, epoch_T, epoch_H, P𝑮
end

########## Run sets of models ####################
function runAllRaw(; n::Int=1000, α::Float64=0.5, θ::Float64=5.0 )

    habit = runHabit(N=n,α=α,θ=θ)
    MF = runMF(N=n,α=α,θ=θ)
    MB = runMB(N=n,α=α,θ=θ)
    GD = runGD(N=n,α=α,θ=θ)

    return habit, MF, MB, GD
end
function runAllBlend(; n::Int=1000, α::Float64=0.5, θ::Float64=5.0 )

    HWV = runHWV(N=n,α=α,θ=θ)

    return  HWV
end
