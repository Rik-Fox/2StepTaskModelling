import Random

############# ASK THE ENVIRON ###############

#### DAW
function agentCtrller(agent::DecisionTree, α::T, β1::T, β2::T, λ::T, ηₜ::T, ηᵣ::T) where T <: Float64

    agent_C = deepcopy(agent) ## cached controller
    agent_TV = deepcopy(agent) ## tree view controller

    π = softMax([agent.state.Q.A1, agent.state.Q.A2], β = β1)
    # select action using softmax policy and set eligbility
    if π > rand()
        actn = "A1"
        agent_C.state.e.A1 = 1.0
        agent_TV.state.e.A1 = 1.0
    else
        actn = "A2"
        agent_C.state.e.A2 = 1.0
        agent_TV.state.e.A2 = 1.0
    end
    ## take action in enivron and update model and action values
    μ, Rwd = askEnviron("ξ", actn)
    agentUpdate(agent_C, actn, μ, α, λ)
    ### set α = 1.0 as direct update
    agentUpdate(agent_TV, actn, μ, 1.0, λ)
    agent.state = rwdUpdate(agent.state, actn, Rwd, ηᵣ)
    agent.state.T = transitionUpdate(agent.state.T, actn, μ, ηₜ)

    if μ
        stNm = "μ"
        π1 = softMax([agent.μ.state.Q.A1, agent.μ.state.Q.A2], β = β2)
        if π1 > rand()
            actn1 = "A1"
            agent_C.μ.state.e.A1 = 1.0
            agent_TV.μ.state.e.A1 = 1.0
            ### as direct update
            δ = agent.μ.state.R.A1
        else
            actn1 = "A2"
            agent_C.μ.state.e.A2 = 1.0
            agent_TV.μ.state.e.A2 = 1.0
            δ = agent.μ.state.R.A2
        end
        μ1, Rwd1 = askEnviron(stNm, actn1)
        ## can bring replace trace up one scope with direct update
        replacetraceUpdate(agent_C, λ, α, Rwd1)
        replacetraceUpdate(agent_TV, λ, 1.0, δ)
        agent.μ.state = rwdUpdate(agent.μ.state, actn1, Rwd1, ηᵣ)
        agent.μ.state.T = transitionUpdate(agent.μ.state.T, actn, ηₜ)
    else
        stNm = "ν"
        π1 = softMax([agent.ν.state.Q.A1, agent.ν.state.Q.A2], β = β2)
        if π1 > rand()
            actn1 = "A1"
            agent_C.ν.state.e.A1 = 1.0
            agent_TV.ν.state.e.A1 = 1.0
            δ = agent.ν.state.R.A1
        else
            actn1 = "A2"
            agent_C.ν.state.e.A2 = 1.0
            agent_TV.ν.state.e.A2 = 1.0
            δ = agent.ν.state.R.A2
        end
        μ1, Rwd1 = askEnviron(stNm, actn1)
        replacetraceUpdate(agent_C, λ, α, Rwd1)
        replacetraceUpdate(agent_TV, λ, 1.0, δ)
        agent.ν.state = rwdUpdate(agent.ν.state, actn1, Rwd1, ηᵣ)
        agent.ν.state.T = transitionUpdate(agent.ν.state.T, actn, ηₜ)
    end

    actn = actn, μ, actn1, π1
    #return original agent, both controller values and relevant environ data
    return agent, Rwd1, actn, π, agent_C, agent_TV
end

##########################################################
## Miller
function agentCtrller(agent::DecisionTree, α::T, β::T, ηₜ::T, ηᵣ::T, D::Array{T,1}) where T <: Float64
    ## uses D, i.e. variance related proportion of h and Q, for softmax policy
    π = softMax([D[1], D[2]], β = β)

    if π > rand()
        actn = "A1"
    else
        actn = "A2"
    end

    μ, Rwd = askEnviron("ξ", actn)
    agentUpdate(agent, actn, μ)
    agent.state.h = habitUpdate(agent.state.h, actn, α)
    agent.state = rwdUpdate(agent.state, actn, Rwd, ηᵣ)
    agent.state.T = transitionUpdate(agent.state.T, actn, μ, ηₜ)
    if μ
        π1 = softMax([D[3], D[4]], β = β)

        if π1 > rand()
            actn1 = "A1"
        else
            actn1 = "A2"
        end
        μ1, Rwd1 = askEnviron("μ", actn1)
        actn1 == "A1" ? agent.μ.state.Q.A1 = agent.μ.state.R.A1 : agent.μ.state.Q.A2 = agent.μ.state.R.A2
        agent.μ.state.h = habitUpdate(agent.μ.state.h, actn1, α)
        agent.μ.state = rwdUpdate(agent.μ.state, actn1, Rwd1, ηᵣ)
        agent.μ.state.T = transitionUpdate(agent.μ.state.T, actn1, ηₜ)

    else
        π1 = softMax([D[5], D[6]], β = β)

        if π1 > rand()
            actn1 = "A1"
        else
            actn1 = "A2"
        end
        μ1, Rwd1 = askEnviron("ν", actn1)
        ## direct update for model based Q/action values
        actn1 == "A1" ? agent.ν.state.Q.A1 = agent.ν.state.R.A1 : agent.ν.state.Q.A2 = agent.ν.state.R.A2
        agent.ν.state.h = habitUpdate(agent.ν.state.h, actn1, α)
        agent.ν.state = rwdUpdate(agent.ν.state, actn1, Rwd1, ηᵣ)
        agent.ν.state.T = transitionUpdate(agent.ν.state.T, actn1, ηₜ)
    end
    actn = actn, μ, actn1, π1
    ### agent holds both controllers internally in this model method
    return agent, Rwd1, actn, π

end

############################################################
## Dezfouli
function agentCtrller(agent::DecisionTree, α1::T, α2::T, β1::T, β2::T, λ::T, ηₜ::T, κ::T, prev_actns::Array, w::T) where T <: Float64

    agent_V = deepcopy(agent) ## model based controller
    agent_Q = deepcopy(agent) ## model free controller
    # κ is a stickiness value, i.e. bigger κ = bigger chance of repeating the last action
    π = softMax([agent.state.Q.A1, agent.state.Q.A2], κ, prev_actns[1], β = β1)
    if π > rand()
        actn = "A1"
        agent_Q.state.e.A1 = 1.0
    else
        actn = "A2"
        agent_Q.state.e.A2 = 1.0
    end
    μ, Rwd = askEnviron("ξ", actn)
    agent_V = agentUpdate(agent_V, actn, μ)
    agent_Q = agentUpdate(agent_Q, actn, μ, α1, λ)
    agent.state.T = transitionUpdate(agent.state.T, actn, μ, ηₜ)
    if μ
        π1 = softMax([agent.μ.state.Q.A1, agent.μ.state.Q.A2], κ, prev_actns[2], β = β2)
        if π1 > rand()
            actn1 = "A1"
            agent_Q.μ.state.e.A1 = 1.0
        else
            actn1 = "A2"
            agent_Q.μ.state.e.A2 = 1.0
        end
        μ1, Rwd1 = askEnviron("μ", actn1)
        actn1 == "A1" ? agent_V.μ.state.Q.A1 = Rwd1 : agent_V.μ.state.Q.A2 = Rwd1
        replacetraceUpdate(agent_Q, λ, α2, Rwd1)
        agent.μ.state.T = transitionUpdate(agent.μ.state.T, actn1, ηₜ)
    else
        π1 = softMax([agent.ν.state.Q.A1, agent.ν.state.Q.A2], κ, prev_actns[2], β = β2)
        if π1 > rand()
            actn1 = "A1"
            agent_Q.ν.state.e.A1 = 1.0
        else
            actn1 = "A2"
            agent_Q.ν.state.e.A2 = 1.0
        end
        μ1, Rwd1 = askEnviron("ν", actn1)
        actn1 == "A1" ? agent_V.ν.state.Q.A1 = Rwd1 : agent_V.ν.state.Q.A2 = Rwd1
        replacetraceUpdate(agent_Q, λ, α2, Rwd1)
        agent.ν.state.T = transitionUpdate(agent.ν.state.T, actn1, ηₜ)
    end
    ### proportional arbitration happens here instead of higher scope runSim for this model, inconsistent with other method structure FIX for continuity, but others performs correctly
    Vᵧ = [agent_V.state.Q.A1, agent_V.state.Q.A2, agent_V.μ.state.Q.A1, agent_V.μ.state.Q.A2, agent_V.ν.state.Q.A1, agent_V.ν.state.Q.A2]
    Qₕ = [agent_Q.state.Q.A1, agent_Q.state.Q.A2, agent_Q.μ.state.Q.A1, agent_Q.μ.state.Q.A2, agent_Q.ν.state.Q.A1, agent_Q.ν.state.Q.A2]
    V = w .* Vᵧ .+ (1 - w) .* Qₕ

    agent.state.Q.A1, agent.state.Q.A2, agent.μ.state.Q.A1, agent.μ.state.Q.A2, agent.ν.state.Q.A1, agent.ν.state.Q.A2 = V[1], V[2], V[3], V[4], V[5], V[6]
    agent.state.e.A1, agent.state.e.A2, agent.μ.state.e.A1, agent.μ.state.e.A2, agent.ν.state.e.A1, agent.ν.state.e.A2 = agent_Q.state.e.A1, agent_Q.state.e.A2, agent_Q.μ.state.e.A1, agent_Q.μ.state.e.A2, agent_Q.ν.state.e.A1, agent_Q.ν.state.e.A2

    actn = actn, μ, actn1, π1

    return agent, Rwd1, actn, π
end

####### ASK THE DATA ###############

### DAW
function agentCtrller(agent::DecisionTree, α::T, β1::T, β2::T, λ::T, ηₜ::T, ηᵣ::T, data::Array{Bool,1}) where T <: Float64

    agent_C = deepcopy(agent)
    agent_TV = deepcopy(agent)

    π = softMax([agent.state.Q.A1, agent.state.Q.A2], β = β1)
    ## uses data to select action, π still calculated for comparing the softmax policy to the taken actions
    if data[1]
        actn = "A1"
        agent_C.state.e.A1 = 1.0
        agent_TV.state.e.A1 = 1.0
    else
        actn = "A2"
        agent_C.state.e.A2 = 1.0
        agent_TV.state.e.A2 = 1.0
    end

    μ, Rwd = askEnviron("ξ", data[1:2])
    agentUpdate(agent_C, actn, μ, α, λ)
    agentUpdate(agent_TV, actn, μ, 1.0, λ)
    agent.state = rwdUpdate(agent.state, actn, Rwd, ηᵣ)
    agent.state.T = transitionUpdate(agent.state.T, actn, μ, ηₜ)

    if μ
        stNm = "μ"
        π1 = softMax([agent.μ.state.Q.A1, agent.μ.state.Q.A2], β = β2)
        if data[3]
            actn1 = "A1"
            agent_C.μ.state.e.A1 = 1.0
            agent_TV.μ.state.e.A1 = 1.0
            δ = agent.μ.state.R.A1
        else
            actn1 = "A2"
            agent_C.μ.state.e.A2 = 1.0
            agent_TV.μ.state.e.A2 = 1.0
            δ = agent.μ.state.R.A2
        end
        μ1, Rwd1 = askEnviron(stNm, data[3:4])
        replacetraceUpdate(agent_C, λ, α, Rwd1)
        replacetraceUpdate(agent_TV, λ, 1.0, δ)
        agent.μ.state = rwdUpdate(agent.μ.state, actn1, Rwd1, ηᵣ)
        agent.μ.state.T = transitionUpdate(agent.μ.state.T, actn, ηₜ)
    else
        stNm = "ν"
        π1 = softMax([agent.ν.state.Q.A1, agent.ν.state.Q.A2], β = β2)
        if data[3]
            actn1 = "A1"
            agent_C.ν.state.e.A1 = 1.0
            agent_TV.ν.state.e.A1 = 1.0
            δ = agent.ν.state.R.A1
        else
            actn1 = "A2"
            agent_C.ν.state.e.A2 = 1.0
            agent_TV.ν.state.e.A2 = 1.0
            δ = agent.ν.state.R.A2
        end
        μ1, Rwd1 = askEnviron(stNm, data[3:4])
        replacetraceUpdate(agent_C, λ, α, Rwd1)
        replacetraceUpdate(agent_TV, λ, 1.0, δ)
        agent.ν.state = rwdUpdate(agent.ν.state, actn1, Rwd1, ηᵣ)
        agent.ν.state.T = transitionUpdate(agent.ν.state.T, actn, ηₜ)
    end

    actn = actn, μ, actn1, π1

    return agent, Rwd1, actn, π, agent_C, agent_TV
end

##########################################################

## Miller
function agentCtrller(agent::DecisionTree, α::T, β::T, ηₜ::T, ηᵣ::T, D::Array{T,1}, data::Array{Bool,1}) where T <: Float64


    π = softMax([D[1], D[2]], β = β)

    if data[1]
        actn = "A1"
    else
        actn = "A2"
    end

    μ, Rwd = askEnviron("ξ", data[1:2])
    agentUpdate(agent, actn, μ)
    agent.state.h = habitUpdate(agent.state.h, actn, α)
    agent.state = rwdUpdate(agent.state, actn, Rwd, ηᵣ)
    agent.state.T = transitionUpdate(agent.state.T, actn, μ, ηₜ)
    if μ
        π1 = softMax([D[3], D[4]], β = β)

        if data[3]
            actn1 = "A1"
        else
            actn1 = "A2"
        end
        μ1, Rwd1 = askEnviron("μ", data[3:4])
        actn1 == "A1" ? agent.μ.state.Q.A1 = agent.μ.state.R.A1 : agent.μ.state.Q.A2 = agent.μ.state.R.A2
        agent.μ.state.h = habitUpdate(agent.μ.state.h, actn1, α)
        agent.μ.state = rwdUpdate(agent.μ.state, actn1, Rwd1, ηᵣ)
        agent.μ.state.T = transitionUpdate(agent.μ.state.T, actn1, ηₜ)

    else
        π1 = softMax([D[5], D[6]], β = β)

        if data[3]
            actn1 = "A1"
        else
            actn1 = "A2"
        end
        μ1, Rwd1 = askEnviron("ν", data[3:4])
        actn1 == "A1" ? agent.ν.state.Q.A1 = agent.ν.state.R.A1 : agent.ν.state.Q.A2 = agent.ν.state.R.A2
        agent.ν.state.h = habitUpdate(agent.ν.state.h, actn1, α)
        agent.ν.state = rwdUpdate(agent.ν.state, actn1, Rwd1, ηᵣ)
        agent.ν.state.T = transitionUpdate(agent.ν.state.T, actn1, ηₜ)
    end
    actn = actn, μ, actn1, π1

    return agent, Rwd1, actn, π
end


#############################################################
## Dezfouli
function agentCtrller(agent::DecisionTree, α1::T, α2::T, β1::T, β2::T, λ::T, ηₜ::T, κ::T, prev_actns::Array{String,1}, w::T, data::Array{Bool,1}) where T <: Float64
    agent_V = deepcopy(agent)
    agent_Q = deepcopy(agent)

    π = softMax([agent.state.Q.A1, agent.state.Q.A2], κ, prev_actns[1], β = β1,)
    if data[1]
        actn = "A1"
        agent_Q.state.e.A1 = 1.0
    else
        actn = "A2"
        agent_Q.state.e.A2 = 1.0
    end
    μ, Rwd = askEnviron("ξ", data[1:2])
    agent_V = agentUpdate(agent_V, actn, μ)
    agent_Q = agentUpdate(agent_Q, actn, μ, α1, λ)
    agent.state.T = transitionUpdate(agent.state.T, actn, μ, ηₜ)
    if μ
        π1 = softMax([agent.μ.state.Q.A1, agent.μ.state.Q.A2], κ, prev_actns[2], β = β2)
        if data[3]
            actn1 = "A1"
            agent_Q.μ.state.e.A1 = 1.0
        else
            actn1 = "A2"
            agent_Q.μ.state.e.A2 = 1.0
        end
        μ1, Rwd1 = askEnviron("μ", data[3:4])
        actn1 == "A1" ? agent_V.μ.state.Q.A1 = Rwd1 : agent_V.μ.state.Q.A2 = Rwd1
        replacetraceUpdate(agent_Q, λ, α2, Rwd1)
        agent.μ.state.T = transitionUpdate(agent.μ.state.T, actn1, ηₜ)
    else
        π1 = softMax([agent.ν.state.Q.A1, agent.ν.state.Q.A2], κ, prev_actns[2], β = β2)
        if data[3]
            actn1 = "A1"
            agent_Q.ν.state.e.A1 = 1.0
        else
            actn1 = "A2"
            agent_Q.ν.state.e.A2 = 1.0
        end
        μ1, Rwd1 = askEnviron("ν", data[3:4])
        actn1 == "A1" ? agent_V.ν.state.Q.A1 = Rwd1 : agent_V.ν.state.Q.A2 = Rwd1
        replacetraceUpdate(agent_Q, λ, α2, Rwd1)
        agent.ν.state.T = transitionUpdate(agent.ν.state.T, actn1, ηₜ)
    end

    Vᵧ = [agent_V.state.Q.A1, agent_V.state.Q.A2, agent_V.μ.state.Q.A1, agent_V.μ.state.Q.A2, agent_V.ν.state.Q.A1, agent_V.ν.state.Q.A2]
    Qₕ = [agent_Q.state.Q.A1, agent_Q.state.Q.A2, agent_Q.μ.state.Q.A1, agent_Q.μ.state.Q.A2, agent_Q.ν.state.Q.A1, agent_Q.ν.state.Q.A2]
    V = w .* Vᵧ .+ (1 - w) .* Qₕ

    agent.state.Q.A1, agent.state.Q.A2, agent.μ.state.Q.A1, agent.μ.state.Q.A2, agent.ν.state.Q.A1, agent.ν.state.Q.A2 = V[1], V[2], V[3], V[4], V[5], V[6]
    agent.state.e.A1, agent.state.e.A2, agent.μ.state.e.A1, agent.μ.state.e.A2, agent.ν.state.e.A1, agent.ν.state.e.A2 = agent_Q.state.e.A1, agent_Q.state.e.A2, agent_Q.μ.state.e.A1, agent_Q.μ.state.e.A2, agent_Q.ν.state.e.A1, agent_Q.ν.state.e.A2

    actn = actn, μ, actn1, π1

    return agent, Rwd1, actn, π
end
