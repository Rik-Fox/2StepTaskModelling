######### Base model runners

## ask the environ
#model free
function runSim(agent::DecisionTree{State{String, T1, T1, T2, T1}}, n::T0; α::T=0.5, β::T=5.0) where {T<:Float64, T0<:Int64, T1<:Actions, T2<:Nothing}
    epoch = zeros(6,n)
    epochT = zeros(6,n)
    Rwd = zeros(n)
    p1 = zeros(n)
    p2 = zeros(n)
    for i=1:n
        itr = agentCtrller(agent,α=α,β=β)
        Rwd[i],p1[i],p2[i] = itr[2],itr[4],itr[3][4]
        epoch[:,i] = [agent.state.Q.A1, agent.state.Q.A2, agent.μ.state.Q.A1, agent.μ.state.Q.A2, agent.ν.state.Q.A1, agent.ν.state.Q.A2]
        epochT[:,i] = [agent.state.T.A1, agent.state.T.A2, agent.μ.state.T.A1, agent.μ.state.T.A2, agent.ν.state.T.A1, agent.ν.state.T.A2]
    end

    return agent, (Rwd, p1, p2), (epoch, epochT)
end
# model based
function runSim(agent::DecisionTree{State{String, T1, T1, T2, T1}}, n::T0, M::T0; α::T=0.5, β::T=5.0) where {T<:Float64, T0<:Int64, T1<:Actions, T2<:Nothing}
    epoch = zeros(6,n)
    epoch_T = zeros(6,n)
    Rwd = zeros(n)
    p1 = zeros(n)
    p2 = zeros(n)
    for i=1:n
        itr = agentCtrller(agent,M,α=α,β=β)
        Rwd[i],p1[i],p2[i] = itr[2],itr[4],itr[3][4]
        epoch[:,i] = [agent.state.Q.A1, agent.state.Q.A2, agent.μ.state.Q.A1, agent.μ.state.Q.A2, agent.ν.state.Q.A1, agent.ν.state.Q.A2]
        epoch_T[:,i] = [agent.state.T.A1, agent.state.T.A2, agent.μ.state.T.A1, agent.μ.state.T.A2, agent.ν.state.T.A1, agent.ν.state.T.A2]
    end

    return agent, (Rwd, p1, p2), (epoch, epochT)
end

### ask the data
#model based
function runSim(agent::DecisionTree{State{String, T1, T1, T2, T1}}, data::Array{Bool,2}, M::Int64; α::T=0.5, β::T=5.0) where {T<:Float64, T1<:Actions, T2<:Nothing}
    n = length(data[:,1])
    epoch = zeros(6,n)
    epochT = zeros(6,n)
    Rwd = zeros(n)
    p1 = zeros(n)
    p2 = zeros(n)
    for i=1:n
        itr = agentCtrller(agent,data[i,:],M,α=α,β=β)
        Rwd[i],p1[i],p2[i] = itr[2],itr[4],itr[3][4]
        epoch[:,i] = [agent.state.Q.A1, agent.state.Q.A2, agent.μ.state.Q.A1, agent.μ.state.Q.A2, agent.ν.state.Q.A1, agent.ν.state.Q.A2]
        epochT[:,i] = [agent.state.T.A1, agent.state.T.A2, agent.μ.state.T.A1, agent.μ.state.T.A2, agent.ν.state.T.A1, agent.ν.state.T.A2]
    end

    return agent, (Rwd, p1, p2), (epoch, epochT)
end
# model free
function runSim(agent::DecisionTree{State{String, T1, T1, T2, T1}}, data::Array{Bool,2}; α::T=0.5, β::T=5.0) where {T<:Float64, T1<:Actions, T2<:Nothing}



    n = length(data[:,1])
    epoch = zeros(6,n)
    epochT = zeros(6,n)
    Rwd = zeros(n)
    p1 = zeros(n)
    p2 = zeros(n)
    for i=1:n
        itr = agentCtrller(agent,data[i,:],α=α,β=β)
        Rwd[i],p1[i],p2[i] = itr[2],itr[4],itr[3][4]
        epoch[:,i] = [agent.state.Q.A1, agent.state.Q.A2, agent.μ.state.Q.A1, agent.μ.state.Q.A2, agent.ν.state.Q.A1, agent.ν.state.Q.A2]
        epochT[:,i] = [agent.state.T.A1, agent.state.T.A2, agent.μ.state.T.A1, agent.μ.state.T.A2, agent.ν.state.T.A1, agent.ν.state.T.A2]
    end

    return agent, (Rwd, p1, p2), (epoch, epochT)
end

###### ask the Environ
#### habit
# function runSim(agent::DecisionTree{State{String, T2, T2, T1, T1}}, n::Int64; α::T=0.5, β::T=5.0) where {T<:AbstractFloat, T1<:Actions, T2<:Nothing}
#     epoch = zeros(6,n)
#     Rwd = zeros(n)
#     p1 = zeros(n)
#     p2 = zeros(n)
#     for i=1:n
#         itr = agentCtrller(agent,α=α,β=β)
#         Rwd[i],p1[i],p2[i] = itr[2],itr[4],itr[3][4]
#         epoch[:,i] = [agent.state.h.A1, agent.state.h.A2, agent.μ.state.h.A1, agent.μ.state.h.A2, agent.ν.state.h.A1, agent.ν.state.h.A2]
#     end
#
#     return agent, (Rwd, p1, p2), epoch
# end


## no transition Probs
# function runSim(agent::DecisionTree{State{String, T1, T2, T2, T1}}, n::Int64; α::T=0.5, β::T=5.0) where {T<:AbstractFloat, T1<:Actions, T2<:Nothing}
#     epoch = zeros(6,n)
#     Rwd = zeros(n)
#     p1 = zeros(n)
#     p2 = zeros(n)
#     for i=1:n
#         itr = agentCtrller(agent,α=α,β=β)
#         Rwd[i],p1[i],p2[i] = itr[2],itr[4],itr[3][4]
#         epoch[:,i] = [agent.state.Q.A1, agent.state.Q.A2, agent.μ.state.Q.A1, agent.μ.state.Q.A2, agent.ν.state.Q.A1, agent.ν.state.Q.A2]
#     end
#
#     return agent, (Rwd, p1, p2), epoch
# end


######## ask the data


## habit
# function runSim(agent::DecisionTree{State{String, T2, T2, T1, T1}}, data::Array{Bool,2}; α::T=0.5, β::T=5.0) where {T<:AbstractFloat, T1<:Actions, T2<:Nothing}
#     n = length(data[:,1])
#     epoch_H = zeros(6,n)
#     Rwd = zeros(n)
#     p1 = zeros(n)
#     p2 = zeros(n)
#     for i=1:n
#         itr = agentCtrller(agent,α=α,β=β)
#         Rwd[i],p1[i],p2[i] = itr[2],itr[4],itr[3][4]
#         epoch_H[:,i] = [agent.state.h.A1, agent.state.h.A2, agent.μ.state.h.A1, agent.μ.state.h.A2, agent.ν.state.h.A1, agent.ν.state.h.A2]
#     end
#
#     return agent, (Rwd, p1, p2), Nothing(), Nothing(), epoch_H
# end

## no transition Probs
# function runSim(agent::DecisionTree{State{String, T1, T2, T2, T1}}, data::Array{Bool,2}; α::T=0.5,β::T=5.0) where {T<:AbstractFloat, T1<:Actions, T2<:Nothing}
#     n = length(data[:,1])
#     epoch = zeros(6,n)
#     Rwd = zeros(n)
#     p1 = zeros(n)
#     p2 = zeros(n)
#     for i=1:n
#         itr = agentCtrller(agent,data[i,:],α=α,β=β)
#         Rwd[i],p1[i],p2[i] = itr[2],itr[4],itr[3][4]
#         epoch[:,i] = [agent.state.Q.A1, agent.state.Q.A2, agent.μ.state.Q.A1, agent.μ.state.Q.A2, agent.ν.state.Q.A1, agent.ν.state.Q.A2]
#     end
#
#     return agent, (Rwd, p1, p2), epoch
# end
