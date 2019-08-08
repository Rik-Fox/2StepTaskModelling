using Random, Plots, CSV, DataFrames
pyplot()
push!(LOAD_PATH, pwd())
using AgentTree

#agent = buildAgent(2,TM=true,hm=true)

############# Plot values of an agent throughout an epoch #######################################
function plotSim(f::Function; data::Union{AbstractArray,Nothing}=Nothing(), ana::Union{Matrix,Bool}=false, N::Int=1000, α::Float64=0.5, θ::Float64=5.0 )
    if typeof(data) != Nothing
        N = length(data[:,1])
    end
    Model = f(n=N,data=data,α=α,θ=θ)
    plt_Q,bar_Q,plt_T,bar_T,plt_h,bar_h,plt_arb = Nothing(),Nothing(),Nothing(),Nothing(),Nothing(),Nothing(),Nothing()
    m = "$f"[4:end]

    if Model[2] != nothing
        if ana == true
            anaQ = zeros(8,N)
            for i=1:N-1
                anaQ[1,i+1] = (1-α)*anaQ[1,i] + α*(0.7*anaQ[3,i]+0.3*anaQ[4,i])
                anaQ[2,i+1] = (1-α)*anaQ[2,i] + α*(0.3*anaQ[3,i]+0.7*anaQ[4,i])
                anaQ[3,i+1] = (1-α)*anaQ[3,i] + α*(findmax([anaQ[5,i] anaQ[6,i]])[1])
                anaQ[4,i+1] = (1-α)*anaQ[4,i] + α*(findmax([anaQ[7,i] anaQ[8,i]])[1])
                anaQ[5,i+1] = (1-α)*anaQ[5,i] + α*0.8
                anaQ[6,i+1] = (1-α)*anaQ[6,i] + α*0.0
                anaQ[7,i+1] = (1-α)*anaQ[7,i] + α*0.2
                anaQ[8,i+1] = (1-α)*anaQ[8,i] + α*0.0
            end
        elseif typeof(ana) == Matrix{Float64}
            anaQ = zeros(8,N)
            anaQ[5:8,:] = ana'
            for i=1:N-1
                anaQ[1,i+1] = (1-α)*anaQ[1,i] + α*(0.7*anaQ[3,i]+0.3*anaQ[4,i])
                anaQ[2,i+1] = (1-α)*anaQ[2,i] + α*(0.3*anaQ[3,i]+0.7*anaQ[4,i])
                anaQ[3,i+1] = (1-α)*anaQ[3,i] + α*(findmax([anaQ[5,i] anaQ[6,i]])[1])
                anaQ[4,i+1] = (1-α)*anaQ[4,i] + α*(findmax([anaQ[7,i] anaQ[8,i]])[1])
            end
        else
            anaQ = zeros(4,N)
        end

        plt_Q = plot(Model[2][1,:],label="ξ.A1",color="blue",ylims = (0, 1))
        plot!(Model[2][2,:],label="ξ.A2",color="orange")
        #plot!(Model[2][3,:],label="μ.A1",color="green")
        #plot!(Model[2][4,:],label="μ.A2")
        #plot!(Model[2][5,:],label="ν.A1",color="magenta")
        #plot!(Model[2][6,:],label="ν.A2")
        plot!(anaQ[1,:],label="Analytic ξ.A1",color="blue",linestyle=:dash)
        plot!(anaQ[2,:],label="Analytic ξ.A2",color="orange",linestyle=:dash)
        #plot!(anaQ[3,:],label="Analytic μ.A1",color="green",linestyle=:dash)
        #plot!(anaQ[4,:],label="Analytic ν.A1",color="magenta",linestyle=:dash)

        title!("$m Time Series of Q values")
        xaxis!("Number of iterations")
        yaxis!("Q(s,a)")

        bar_Q = bar([Model[1].state.Q.A1, Model[1].state.Q.A2, Model[1].μ.state.Q.A1, Model[1].μ.state.Q.A2, Model[1].ν.state.Q.A1, Model[1].ν.state.Q.A2],legend=false,ylims = (0, 1));
        title!("$m Final Q values")
        xaxis!("Action(s) Taken")
        xticks!((1:6),["ξ.A1", "ξ.A2", "μ.A1", "μ.A2", "ν.A1", "ν.A2"])
        yaxis!("Q(s,a)")
    end

    if Model[1].state.T != nothing

        plt_T = plot(Model[3][1,:],label="ξ.A1",color="blue",ylims = (0, 1))
        plot!(Model[3][2,:],label="ξ.A2",color="orange")
        plot!(Model[3][3,:],label="μ.A1",color="green")
        plot!(Model[3][4,:],label="μ.A2")
        plot!(Model[3][5,:],label="ν.A1",color="magenta")
        plot!(Model[3][6,:],label="ν.A2")
        title!("$m Time Series of Transition Model")

        xaxis!("Number of iterations")
        yaxis!("T(s,a,s')")

        bar_T = bar([Model[1].state.T.A1, Model[1].state.T.A2, Model[1].μ.state.T.A1, Model[1].μ.state.T.A2, Model[1].ν.state.T.A1, Model[1].ν.state.T.A2],legend=false,ylims = (0, 1));
        title!("$m Final Transition Model")
        xaxis!("Action(s) Taken")
        xticks!((1:6),["ξ.A1", "ξ.A2", "μ.A1", "μ.A2", "ν.A1", "ν.A2"])
        yaxis!("T(s,a,s')")
    end

    if Model[4] != nothing
        plt_h = plot(Model[4][1,:],label="ξ.A1",color="blue",ylims = (0, 1))
        plot!(Model[4][2,:],label="ξ.A2",color="orange")
        plot!(Model[4][3,:],label="μ.A1",color="green")
        plot!(Model[4][4,:],label="μ.A2")
        plot!(Model[4][5,:],label="ν.A1",color="magenta")
        plot!(Model[4][6,:],label="ν.A2")

        title!("$m Time Series of values")
        xaxis!("Number of iterations")
        yaxis!("h(s,a)")

        bar_h = bar([Model[1].state.h.A1, Model[1].state.h.A2, Model[1].μ.state.h.A1, Model[1].μ.state.h.A2, Model[1].ν.state.h.A1, Model[1].ν.state.h.A2],legend=false,ylims = (0, 1));
        title!("$m Final Habit values")
        xaxis!("Action(s) Taken")
        xticks!((1:6),["ξ.A1", "ξ.A2", "μ.A1", "μ.A2", "ν.A1", "ν.A2"])
        yaxis!("h(s,a)")
    end
    if Model[1].state.T != nothing && Model[4] != nothing
        plt_arb = plot(Model[5],ylims = (0, 1))
        title!("Probability over time of Goal Directed Controller being chosen")
        xaxis!("Number of iterations")
        yaxis!("Probability of Goal Directed Controller being chosen")
    end

    return Model,plt_Q,bar_Q,plt_T,bar_T,plt_h,bar_h,plt_arb,anaQ
end

function plotData(exData::AbstractArray,exRwdProb::AbstractArray; α::Float64=0.5, θ::Float64=5.0)
    habitSimResults = plotSim(runHabit,data=exData,ana=exRwdProb,α=α)
    MFSimResults = plotSim(runMF,data=exData,ana=exRwdProb,α=α)
    MBSimResults = plotSim(runMB,data=exData,ana=exRwdProb,α=α)
    GDSimResults = plotSim(runGD,data=exData,ana=exRwdProb,α=α)
    HWVSimResults = plotSim(runHWV,data=exData,ana=exRwdProb,α=α)

    plth = habitSimResults[2]
    pltf = MFSimResults[2]
    pltb = MBSimResults[2]
    pltg = GDSimResults[2]
    pltw = HWVSimResults[2]

    pltQ = plot(plth,pltf,pltb,pltg,pltw,size=(1000,600),legend=false)

    pltb = MBSimResults[4]
    pltg = GDSimResults[4]
    pltw = HWVSimResults[4]

    pltT = plot(pltb,pltg,pltw,size=(1000,600))

    plth = habitSimResults[6]
    pltw = HWVSimResults[6]

    pltH = plot(plth,pltw,size=(1000,600))

    return pltQ, pltT, pltH
end

######### Run/Plot all Raw controllers for a full epoch ############################################
function runAllRaw(; n::Int=1000, α::Float64=0.5, θ::Float64=5.0 )

    habit = runHabit(N=n,α=α,θ=θ)
    MF = runMF(N=n,α=α,θ=θ)
    MB = runMB(N=n,α=α,θ=θ)

    return habit, MF, MB
end

function plotAllRaw(; n::Int=1000, α::Float64=0.5, θ::Float64=5.0 )
    ## Raw Models
    habit = plotSim(runHabit,N=n,α=α,θ=θ)
    MF = plotSim(runMF,N=n,α=α,θ=θ)
    MB = plotSim(runMB,N=n,α=α,θ=θ)

    return habit, MF, MB
end

######### Run/Plot all Blend controllers for a full epoch ##########################################
function runAllBlend(; n::Int=1000, α::Float64=0.5, θ::Float64=5.0 )

    GD = runGD(N=n,α=α,θ=θ)
    HWV = runHWV(N=n,α=α,θ=θ)

    return GD, HWV
end

function plotAllBlend(; n::Int=1000, α::Float64=0.5, θ::Float64=5.0 )
    GD = plotSim(runGD,N=n,α=α,θ=θ)
    HWV = plotSim(runHWV,N=n,α=α,θ=θ)

    return GD, HWV
end

################## Misc Functions ##################################################################
function softMax(A; θ::Float64=5.0)
    a_idx = findall(x->x==findmax(A)[1], A)[1]
    a = A[a_idx][1]
    p = exp(θ*a)/sum(exp.(θ*A))
    return p, a_idx
end

function rwd(p)
    if rand() < p
        return 1.0
    else
        return 0.0
    end
end

################### Agent Value Updates ############################################################
function habitUpdate(h::Actions,actn::String,α::Float64)

    if actn == "A1"
        h.A1 = (1-α)*h.A1 + α
        h.A2 = (1-α)*h.A2
    elseif actn == "A2"
        h.A1 = (1-α)*h.A1
        h.A2 = (1-α)*h.A2 + α
    else
        throw(ArgumentError("Action argument must be either \"A1\" or \"A2\""))
    end
    return h
end

function QUpdate(Node::DecisionTree, actn::String, μ::Union{Bool,Nothing}, α::Float64)

    if Node.state.name == "ξ"
        μ ? Q_ = findmax([Node.μ.state.Q.A1, Node.μ.state.Q.A2])[1] : Q_ = findmax([Node.ν.state.Q.A1, Node.ν.state.Q.A2])[1]
    elseif Node.state.name == "μ" || Node.state.name == "ν"
        Q_ = Node.state.R
    else
        throw(error("unrecognised state"))
    end

    if actn == "A1"
        Node.state.Q.A1 = (1-α)*Node.state.Q.A1 + α*Q_
    elseif actn == "A2"
        Node.state.Q.A2 = (1-α)*Node.state.Q.A2 + α*Q_
    else
        throw(ArgumentError("action argument must be either \"A1\" or \"A2\""))
    end

    return Node.state.Q
end

function modelledQUpdate(Node::DecisionTree, actn::String, μ::Union{Bool,Nothing}, α::Float64)

    if Node.state.name == "ξ"       # if in base node Qlearn Eq is updated by Qvalue of
                                    # state landed in as R:=0 in this state
        actn == "A1" ? (p = Node.state.T.A1 ; q = 1-(Node.state.T.A1)) : (p = 1-(Node.state.T.A2) ; q = Node.state.T.A2)           # selecting rare and common transition Probabilities
                Q_ = p*findmax([Node.μ.state.Q.A1, Node.μ.state.Q.A2])[1] + q*findmax([Node.ν.state.Q.A1,Node.ν.state.Q.A2])[1] # max action * T
        if μ #actn == "A1" #μ
            Node.state.Q.A1 = (1-α)*Node.state.Q.A1 + α*Q_
        else
            Node.state.Q.A2 = (1-α)*Node.state.Q.A2 + α*Q_
        end
    elseif Node.state.name == "μ" || Node.state.name == "ν"

        Q_ = Node.state.R
        if actn == "A1"
            Node.state.Q.A1 = (1-α)*Node.state.Q.A1 + α*Q_
        else
            Node.state.Q.A2 = (1-α)*Node.state.Q.A2 + α*Q_
        end
    else
        throw(error("unrecognised state"))
    end

    return Node.state.Q
end

function transitionUpdate(Node::DecisionTree,actn::String,μ::Union{Bool,Nothing},α::Float64)
    if Node.state.name == "ξ"
        μ ? (actn == "A1" ? Node.state.T.A1 = (1-α)*Node.state.T.A1 + α : Node.state.T.A2 = (1-α)*Node.state.T.A2) : (actn == "A1" ? Node.state.T.A1 = (1-α)*Node.state.T.A1 : Node.state.T.A2 = (1-α)*Node.state.T.A2 + α)
    elseif Node.state.name == "μ" || Node.state.name == "ν"
        if actn == "A1"
            Node.state.T.A1 = (1-α)*Node.state.T.A1 + α
        elseif actn == "A2"
            Node.state.T.A2 = (1-α)*Node.state.T.A2 + α
        end
    else
        throw(error("unrecognised state"))
    end

    return Node.state.T
end

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

function taskRead(state::String,data::AbstractArray)
    if state == "ξ"
        data[1] ? (data[2] ? μ = false : μ = true) : (data[2] ? μ = true : μ = false)
        R = 0.0
    else
        μ = Nothing()
        R = Int(data[2])
    end

    return μ, R
end

function taskCreateData(agent::DecisionTree;N::Int=150,α::Float64=0.5,θ::Float64=5.0)
    dat = Array{Any,2}(undef,(N,4))
    for i = 1:N
        x = MFCtrl(agent,α=α,θ=θ)
        dat[i,:] = [x[3][1] x[3][2] x[3][3] x[2]]
    end
    Data = [[d=="A1" for d in dat[:,1]] [d for d in dat[:,2]] [d=="A1" for d in dat[:,3]] [d==1.0 for d in dat[:,4]]]

    return Data
end

# test = taskCreateData(agent)
#
# test[:,4]

################### Agent Controllers ##############################################################

function habitFlatCtrl(Node::DecisionTree; data::Union{AbstractArray,Nothing}=Nothing(), θ::Float64=5.0)

    if typeof(Node.state.h) == Nothing
        throw(error("Agent must have habitual modelling enabled, set kwarg \"hm=true\" when created"))
    end

    α=1.0
    if typeof(data) == Nothing
        π, a_idx = softMax([Node.state.h.A1, Node.state.h.A2],θ=θ)
        rv = rand()
        ((π >= rv && a_idx == 1) || (π < rv && a_idx == 2)) ? actn = "A1" : actn = "A2"

        μ, Rwd = taskEval(Node.state.name,actn)

        if Node.state.name == "ξ"
            μ ? habitCtrl(Node.μ,α=α) : habitCtrl(Node.ν,α=α)
        end
    else
        data[1] ? actn = "A1" : actn = "A2"

        μ, Rwd = taskRead(Node.state.name,data[1:2])

        if Node.state.name == "ξ"
            μ ? habitCtrl(Node.μ,data=data[3:4],α=α) : habitCtrl(Node.ν,data=data[3:4],α=α)
        end
    end

    Node.state.R = Rwd
    Node.state.h = habitUpdate(Node.state.h,actn,α)
    Node.state.Q = QUpdate(Node,actn,μ,α)

    return Node, Rwd
end

function habitCtrl(Node::DecisionTree; data::Union{AbstractArray,Nothing}=Nothing(), α::Float64=0.5,θ::Float64=5.0)

    if typeof(Node.state.h) == Nothing
        throw(error("Agent must have habitual modelling enabled, set kwarg \"hm=true\" when created"))
    end

    if typeof(data) == Nothing
        π, a_idx = softMax([Node.state.h.A1, Node.state.h.A2],θ=θ)
        rv = rand()
        ((π >= rv && a_idx == 1) || (π < rv && a_idx == 2)) ? actn = "A1" : actn = "A2"

        μ, Rwd = taskEval(Node.state.name,actn)
        Node.state.R = Rwd
        Node.state.h = habitUpdate(Node.state.h,actn,α)
        #Node.state.Q = QUpdate(Node,actn,μ,α)
        if Node.state.name == "ξ"
            μ ? habitCtrl(Node.μ,α=α) : habitCtrl(Node.ν,α=α)

        end

    else
        data[1] ? actn = "A1" : actn = "A2"

        μ, Rwd = taskRead(Node.state.name,data[1:2])
        Node.state.R = Rwd
        Node.state.h = habitUpdate(Node.state.h,actn,α)
        #Node.state.Q = QUpdate(Node,actn,μ,α)
        if Node.state.name == "ξ"
            μ ? habitCtrl(Node.μ,data=data[3:4],α=α) : habitCtrl(Node.ν,data=data[3:4],α=α)
        end
    end

    return Node, Rwd
end

function MFCtrl(Node::DecisionTree; data::Union{AbstractArray,Nothing}=Nothing(), α::Float64=0.5,θ::Float64=5.0)

    if typeof(data) == Nothing
        π, a_idx = softMax([Node.state.Q.A1, Node.state.Q.A2],θ=θ)
        rv = rand()
        ((π >= rv && a_idx == 1) || (π < rv && a_idx == 2)) ? actn = "A1" : actn = "A2"

        μ, Rwd = taskEval(Node.state.name,actn)
        Node.state.R = Rwd
        Node.state.Q = QUpdate(Node,actn,μ,α)
        dat = actn
        if Node.state.name == "ξ"
            μ ? dat2=MFCtrl(Node.μ,α=α)[3] : dat2=MFCtrl(Node.ν,α=α)[3]
            dat = actn, μ, dat2
        end
    else
        data[1] ? actn = "A1" : actn = "A2"

        μ, Rwd = taskRead(Node.state.name,data[1:2])
        Node.state.R = Rwd
        Node.state.Q = QUpdate(Node,actn,μ,α)

        if Node.state.name == "ξ"
            μ ? MFCtrl(Node.μ,data=data[3:4],α=α) : MFCtrl(Node.ν,data=data[3:4],α=α)
        end
        dat=Nothing()
    end

    return Node, Rwd, dat
end

function MBCtrl(Node::DecisionTree; data::Union{AbstractArray,Nothing}=Nothing(), α::Float64=0.5,θ::Float64=5.0)
    if typeof(Node.state.T) == Nothing
        throw(error("Agent must have Transition modelling enabled, set kwarg \"TM=true\" when created"))
    end



    if typeof(data) == Nothing
        π, a_idx = softMax([Node.state.Q.A1, Node.state.Q.A2],θ=θ)
        rv = rand()
        ((π >= rv && a_idx == 1) || (π < rv && a_idx == 2)) ? actn = "A1" : actn = "A2"

        μ, Rwd = taskEval(Node.state.name,actn)
        Node.state.R = Rwd
        Node.state.Q = modelledQUpdate(Node,actn,μ,α)

        if Node.state.name == "ξ"
            μ ? MBCtrl(Node.μ,α=α) : MBCtrl(Node.ν,α=α)
        end
    else
        data[1] ? actn = "A1" : actn = "A2"

        μ, Rwd = taskRead(Node.state.name,data[1:2])
        Node.state.R = Rwd
        Node.state.Q = modelledQUpdate(Node,actn,μ,α)

        if Node.state.name == "ξ"
            μ ? MBCtrl(Node.μ,data=data[3:4],α=α) : MBCtrl(Node.ν,data=data[3:4],α=α)
        end
    end


    return Node, Rwd
end

function GDCtrl(Node::DecisionTree; data::Union{AbstractArray,Nothing}=Nothing(), α::Float64=0.5, αₜ::Float64=0.1,θ::Float64=5.0)
    if typeof(Node.state.T) == Nothing
        throw(error("Agent must have Transition modelling enabled, set kwarg \"TM=true\" when created"))
    end

    if typeof(data) == Nothing
        π, a_idx = softMax([Node.state.Q.A1, Node.state.Q.A2],θ=θ)
        rv = rand()
        ((π >= rv && a_idx == 1) || (π < rv && a_idx == 2)) ? actn = "A1" : actn = "A2"

        μ, Rwd = taskEval(Node.state.name,actn)

        Node.state.R = Rwd
        Node.state.Q = modelledQUpdate(Node,actn,μ,α)


        if Node.state.name == "ξ"
            μ ? GDCtrl(Node.μ,α=α) : GDCtrl(Node.ν,α=α)
        end
    else

        data[1] ? actn = "A1" : actn = "A2"

        μ, Rwd = taskRead(Node.state.name,data[1:2])

        Node.state.R = Rwd
        Node.state.Q = modelledQUpdate(Node,actn,μ,α)


        if Node.state.name == "ξ"
            μ ? GDCtrl(Node.μ,data=data[3:4],α=α) : GDCtrl(Node.ν,data=data[3:4],α=α)
        end
    end
        Node.state.T = transitionUpdate(Node,actn,μ,αₜ)
    return Node, Rwd
end

############## Functions that run a full Epoch of a Raw Controller ###############################
function runHabit(; agent::DecisionTree=buildAgent(2,hm=true), data::Union{AbstractArray,Nothing}=Nothing(), n::Int=1000, α::Float64=0.5, θ::Float64=5.0 )
    if typeof(data) != Nothing
        n = length(data[:,1])
    end
    epoch_H = zeros(6,n)
    epoch_Q = zeros(6,n)
    Rwd = zeros(n)
    for i=1:n
        d = Nothing()
        if typeof(data) != Nothing
            d=data[i,:]
        end
        Rwd[i] = habitCtrl(agent,data=d,α=α,θ=θ)[2]
        epoch_H[:,i] = [agent.state.h.A1, agent.state.h.A2, agent.μ.state.h.A1, agent.μ.state.h.A2, agent.ν.state.h.A1, agent.ν.state.h.A2]
        epoch_Q[:,i] = [agent.state.Q.A1, agent.state.Q.A2, agent.μ.state.Q.A1, agent.μ.state.Q.A2, agent.ν.state.Q.A1, agent.ν.state.Q.A2]
    end

    return agent,epoch_Q,Nothing(), epoch_H
end

function runMF(; agent::DecisionTree=buildAgent(2), data::Union{AbstractArray,Nothing}=Nothing(), n::Int=1000, α::Float64=0.5, θ::Float64=5.0)
    if typeof(data) != Nothing
        n = length(data[:,1])
    end
    epoch_Q = zeros(6,n)
    Rwd = zeros(n)
    for i=1:n
        d = Nothing()
        if typeof(data) != Nothing
            d=data[i,:]
        end
        Rwd[i] = MFCtrl(agent,data=d,α=α,θ=θ)[2]
        epoch_Q[:,i] = [agent.state.Q.A1, agent.state.Q.A2, agent.μ.state.Q.A1, agent.μ.state.Q.A2, agent.ν.state.Q.A1, agent.ν.state.Q.A2]
    end

    return agent, epoch_Q, Nothing(), Nothing()
end

function runMB(; agent::DecisionTree=buildAgent(2,TM=true), data::Union{AbstractArray,Nothing}=Nothing(), n::Int=1000, α::Float64=0.5, θ::Float64=5.0, TM::AbstractArray = [0.7 0.7 1.0 1.0 1.0 1.0] )
    agent.state.T.A1, agent.state.T.A2, agent.μ.state.T.A1, agent.μ.state.T.A2, agent.ν.state.T.A1, agent.ν.state.T.A2 = TM[1], TM[2], TM[3], TM[4], TM[5], TM[6]
    if typeof(data) != Nothing
        n = length(data[:,1])
    end
    epoch_Q = zeros(6,n)
    epoch_T = zeros(6,n)
    Rwd = zeros(n)
    for i=1:n
        d = Nothing()
        if typeof(data) != Nothing
            d=data[i,:]
        end
        Rwd = MBCtrl(agent,data=d,α=α,θ=θ)[2]
        epoch_Q[:,i] = [agent.state.Q.A1, agent.state.Q.A2, agent.μ.state.Q.A1, agent.μ.state.Q.A2, agent.ν.state.Q.A1, agent.ν.state.Q.A2]
        epoch_T[:,i] = [agent.state.T.A1, agent.state.T.A2, agent.μ.state.T.A1, agent.μ.state.T.A2, agent.ν.state.T.A1, agent.ν.state.T.A2]
    end

    return agent, epoch_Q, epoch_T, Nothing()
end

########### Functions that run Epochs of Models comprised of Multiple Controllers ##################
function runGD(; agent::DecisionTree=buildAgent(2,TM=true), data::Union{AbstractArray,Nothing}=Nothing(), n::Int=1000, α::Float64=0.5, θ::Float64=5.0 )
    if typeof(data) != Nothing
        n = length(data[:,1])
    end
    epoch_Q = zeros(6,n)
    epoch_T = zeros(6,n)
    Rwd = zeros(n)
    for i=1:n
        d = Nothing()
        if typeof(data) != Nothing
            d=data[i,:]
        end
        Rwd = GDCtrl(agent,data=d,α=α,θ=θ)[2]
        epoch_Q[:,i] = [agent.state.Q.A1, agent.state.Q.A2, agent.μ.state.Q.A1, agent.μ.state.Q.A2, agent.ν.state.Q.A1, agent.ν.state.Q.A2]
        epoch_T[:,i] = [agent.state.T.A1, agent.state.T.A2, agent.μ.state.T.A1, agent.μ.state.T.A2, agent.ν.state.T.A1, agent.ν.state.T.A2]
    end

    return agent, epoch_Q, epoch_T, Nothing()
end

function runHWV(; agent::DecisionTree=buildAgent(2,TM=true,hm=true), data::Union{AbstractArray,Nothing}=Nothing(), n::Int=1000, α::Float64=0.5, θ::Float64=5.0 )
    if typeof(data) != Nothing
        n = length(data[:,1])
    end
    # pre allocating data arrays
    epoch_Q = zeros(6,n)
    epoch_T = zeros(6,n)
    epoch_H = zeros(6,n)
    Rwd = zeros(n)

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
            Rwd = GDCtrl(agent,data=d,α=α,θ=θ)[2]
            r𝑮 = (1-α)*r𝑮 + α*Rwd
        else
            Rwd = habitCtrl(agent,data=d,α=α,θ=θ)[2]
            h_avg = sum([agent.state.h.A1, agent.state.h.A2, agent.μ.state.h.A1, agent.μ.state.h.A2, agent.ν.state.h.A1, agent.ν.state.h.A2])/6
        end
        r0 = (1-α)*r0 + α*Rwd
        epoch_Q[:,i] = [agent.state.Q.A1, agent.state.Q.A2, agent.μ.state.Q.A1, agent.μ.state.Q.A2, agent.ν.state.Q.A1, agent.ν.state.Q.A2]
        epoch_T[:,i] = [agent.state.T.A1, agent.state.T.A2, agent.μ.state.T.A1, agent.μ.state.T.A2, agent.ν.state.T.A1, agent.ν.state.T.A2]
        epoch_H[:,i] = [agent.state.h.A1, agent.state.h.A2, agent.μ.state.h.A1, agent.μ.state.h.A2, agent.ν.state.h.A1, agent.ν.state.h.A2]
    end

    return agent, epoch_Q, epoch_T, epoch_H, P𝑮
end
