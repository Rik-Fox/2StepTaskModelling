using Random, Plots
pyplot()
push!(LOAD_PATH, pwd())
using CustomStructs

function buildStepTask(steps::Int;TM::Bool=false,r::Float64=0.0)
    if steps == 1
        Q = Actions(0.0,0.0)
        TM ? T = Actions(0.5,0.5) : T = Nothing()
        h = Actions(0.0,0.0)
        R = 0.0
        Task = DecisionTree(State(Q,T,h,R),round(0.8-r,digits=2),0.0)

    else
        Q = Actions(0.0,0.0)
        TM ? T = Actions(0.5,0.5) : T = Nothing()
        h = Actions(0.0,0.0)
        R = 0.0
        Task = DecisionTree(State(Q,T,h,R), buildStepTask(steps-1,TM=TM), buildStepTask(steps-1,TM=TM,r=0.6))
    end

    return Task
end

function softMax(a, A; θ::Float64=5.0)
    p = exp(θ*a)/sum(exp.(θ*A))
    return p
end

function rwd(p)
    if rand() < p
        return 1.0
    else
        return 0.0
    end
end

function habitUpdate(state::String,Node::DecisionTree,α::Float64;leaf::Bool=false)

    if state == "A1"
        Node.state.h.A1 = (1-α)*Node.state.h.A1 + α
        Node.state.h.A2 = (1-α)*Node.state.h.A2
        leaf ? Node.state.R = Node.state.R=(1-α)*Node.state.R + α*rwd(Node.A1) : Node.state.R=(1-α)*Node.state.R + α*Node.A1.state.R
    elseif state == "A2"
        Node.state.h.A1 = (1-α)*Node.state.h.A1
        Node.state.h.A2 = (1-α)*Node.state.h.A2 + α
        leaf ? Node.state.R=(1-α)*Node.state.R + α*rwd(Node.A2) : Node.state.R=(1-α)*Node.state.R + α*Node.A2.state.R
    else
        throw(ArgumentError("First argument must be either \"A1\" or \"A2\""))
    end
    return Node
end

function transitionUpdate(state::String,T::Actions,α::Float64;swSt::Bool=false)
    if state == "A1"
        swSt == false ? T.A1 = (1-α)*T.A1 + α : T.A2 = (1-α)*T.A2

    elseif state == "A2"
        swSt == false ? T.A2 = (1-α)*T.A2 + α : T.A1 = (1-α)*T.A1
    else
        throw(ArgumentError("First argument must be either \"A1\" or \"A2\""))
    end
    return T
end

function switchActn(switch::Bool, α::Float64, Node::DecisionTree, switchNode::DecisionTree, f::Function)

    if switch == true
        Hold = deepcopy(Node)
        Node = deepcopy(switchNode)
        Node = f(Node,α=α)
        Node = deepcopy(Hold)
    else
        Node = f(Node,α=α)
    end

    return Node
end

function modelledQUpdate(Node::DecisionTree, actn::String, r::Float64, α::Float64; leaf::Bool=false)

    if  leaf == true
        if actn == "A1"
            Node.state.Q.A1 = (1-α)*Node.state.Q.A1 + α*r
            Node.state.R = (1-α)*Node.state.R + α*r
        elseif actn == "A2"
            Node.state.Q.A2 = (1-α)*Node.state.Q.A2 + α*r
            Node.state.R = (1-α)*Node.state.R + α*r
        else
            throw(ArgumentError("action argument must be either \"A1\" or \"A2\""))
        end
    else

        actn == "A1" ? (p = Node.state.T.A1 ; q = 1-(Node.state.T.A1)) : (p = 1-(Node.state.T.A2) ; q = Node.state.T.A2)

        Q_ = p*findmax([Node.A1.state.Q.A2  Node.A1.state.Q.A1])[1] + q*findmax([Node.A2.state.Q.A2  Node.A2.state.Q.A1])[1]

        if actn == "A1"
            Node.state.Q.A1 = (1-α)*Node.state.Q.A1 + α*Q_
            Node.state.R = (1-α)*Node.state.R + α*Node.A1.state.R
        elseif actn == "A2"
            Node.state.Q.A2 = (1-α)*Node.state.Q.A2 + α*Q_
            Node.state.R = (1-α)*Node.state.R + α*Node.A2.state.R
        else
            throw(ArgumentError("action argument must be either \"A1\" or \"A2\""))
        end
    end
end

function QUpdate(Node::DecisionTree, actn::String, r::Float64, α::Float64; leaf::Bool=false)

    if  leaf == true
        if actn == "A1"
            Node.state.Q.A1 = (1-α)*Node.state.Q.A1 + α*r
            Node.state.R = (1-α)*Node.state.R + α*r
        elseif actn == "A2"
            Node.state.Q.A2 = (1-α)*Node.state.Q.A2 + α*r
            Node.state.R = (1-α)*Node.state.R + α*r
        else
            throw(ArgumentError("action argument must be either \"A1\" or \"A2\""))
        end
    else

        if actn == "A1"
            Q_ = findmax([Node.A1.state.Q.A1, Node.A1.state.Q.A2])[1]
            Node.state.Q.A1 = (1-α)*Node.state.Q.A1 + α*Q_
            Node.state.R = (1-α)*Node.state.R + α*Node.A1.state.R
        elseif actn == "A2"
            Q_ = findmax([Node.A2.state.Q.A1, Node.A2.state.Q.A2])[1]
            Node.state.Q.A2 = (1-α)*Node.state.Q.A2 + α*Q_
            Node.state.R = (1-α)*Node.state.R + α*Node.A2.state.R
        else
            throw(ArgumentError("action argument must be either \"A1\" or \"A2\""))
        end
    end

    return Node
end

function habitCtrl(Node::DecisionTree; α::Float64=0.5,θ::Float64=5.0)

    A = [Node.state.h.A1, Node.state.h.A2]
    a_idx = findall(x->x==findmax(A)[1], A)[1]
    π = softMax(A[a_idx][1], A,θ=θ)

    rv = rand()

    if typeof(Node.A1) == Float64 || typeof(Node.A2) == Float64
        ((π >= rv && a_idx == 1) || (π < rv && a_idx == 2)) ? (r=Node.A1;actn = "A1") : (r=Node.A2;actn = "A2")
        habitUpdate(actn,Node,α,leaf=true)
        QUpdate(Node,actn,rwd(r),α,leaf=true)
    else
        if π >= rv && a_idx == 1 || π < rv && a_idx == 2

            rand() < 0.7 ? sw=false : sw=true
            switchActn(sw,α,Node.A1,Node.A2,habitCtrl)
            habitUpdate("A1",Node,α)
            QUpdate(Node,"A1",0.0,α)
        elseif π >= rv && a_idx == 2 || π < rv && a_idx == 1

            rand() < 0.7 ? sw=false : sw=true
            switchActn(sw,α,Node.A2,Node.A1,habitCtrl)
            habitUpdate("A2",Node,α)
            QUpdate(Node,"A2",0.0,α)
        else
            throw(ArgumentError("softmax evaluation is going wrong"))
        end
    end

    return Node
end

function MFCtrl(Node::DecisionTree; α::Float64=0.5,θ::Float64=5.0)

    A = [Node.state.Q.A1, Node.state.Q.A2]
    a_idx = findall(x->x==findmax(A)[1], A)[1]
    π = softMax(A[a_idx][1], A,θ=θ)

    rv = rand()

    if typeof(Node.A1) == Float64 || typeof(Node.A2) == Float64
        ((π >= rv && a_idx == 1) || (π < rv && a_idx == 2)) ? (r=Node.A1;actn = "A1") : (r=Node.A2;actn = "A2")
        QUpdate(Node,actn,rwd(r),α,leaf=true)
    else
        if (π >= rv && a_idx == 1) || (π < rv && a_idx == 2)
            rand() < 0.7 ? sw=false : sw=true
            switchActn(sw,α,Node.A1,Node.A2,MFCtrl)
            QUpdate(Node,"A1",0.0,α)
        elseif (π >= rv && a_idx == 2) || (π < rv && a_idx == 1)
            rand() < 0.7 ? sw=false : sw=true
            switchActn(sw,α,Node.A2,Node.A1,MFCtrl)
            QUpdate(Node,"A2",0.0,α)
        else
            throw(ArgumentError("softmax evaluation is going wrong"))
        end
    end

    return Node
end

function MBCtrl(Node::DecisionTree; α::Float64=0.5,θ::Float64=5.0)
    A = [Node.state.Q.A1, Node.state.Q.A2]
    a_idx = findall(x->x==findmax(A)[1], A)[1]
    π = softMax(A[a_idx][1], A,θ=θ)

    rv = rand()

    if typeof(Node.A1) == Float64 || typeof(Node.A2) == Float64
        ((π >= rv && a_idx == 1) || (π < rv && a_idx == 2)) ? (r=Node.A1;actn = "A1") : (r=Node.A2;actn = "A2")
        modelledQUpdate(Node,actn,rwd(r),α,leaf=true)
    else
        if (π > rv && a_idx == 1) || (π < rv && a_idx == 2)
            actn = "A1"
            rand() < 0.7 ? sw=false : sw=true
            switchActn(sw,α,Node.A1,Node.A2,MBCtrl)
            modelledQUpdate(Node,actn,0.0,α)
        elseif (π > rv && a_idx == 2) || (π < rv && a_idx == 1)
            actn = "A2"
            rand() < 0.7 ? sw=false : sw=true
            switchActn(sw,α,Node.A2,Node.A1,MBCtrl)
            modelledQUpdate(Node,actn,0.0,α)
        end
    end

    return Node
end

function GDCtrl(Node::DecisionTree; α::Float64=0.5, αₜ::Float64=0.05,θ::Float64=5.0)
    A = [Node.state.Q.A1, Node.state.Q.A2]
    a_idx = findall(x->x==findmax(A)[1], A)[1]
    π = softMax(A[a_idx][1], A,θ=θ)

    rv = rand()

    if typeof(Node.A1) == Float64 || typeof(Node.A2) == Float64
        ((π >= rv && a_idx == 1) || (π < rv && a_idx == 2)) ? (r=rwd(Node.A1);actn = "A1") : (r=rwd(Node.A2);actn = "A2")
        transitionUpdate(actn,Node.state.T,αₜ)
        modelledQUpdate(Node,actn,r,α,leaf=true)
    else
        if (π >= rv && a_idx == 1) || (π < rv && a_idx == 2)
            rand() < 0.7 ? (sw=false;actn = "A1") : (sw=true;actn = "A2")
            switchActn(sw,α,Node.A1,Node.A2,GDCtrl)
            transitionUpdate(actn,Node.state.T,αₜ,swSt=sw)
            modelledQUpdate(Node,actn,0.0,α)
        elseif (π >= rv && a_idx == 2) || (π < rv && a_idx == 1)
            rand() < 0.7 ? (sw=false;actn = "A2") : (sw=true;actn = "A1")
            switchActn(sw,α,Node.A2,Node.A1,GDCtrl)
            transitionUpdate(actn,Node.state.T,αₜ,swSt=sw)
            modelledQUpdate(Node,actn,0.0,α)
        end
    end

    return Node
end


function runHWV(; HWV = buildStepTask(2,TM=true), n::Int=1000, α::Float64=0.5, θ::Float64=5.0 )
    # pre allocating data arrays
    epochHWV_Q = zeros(6,n)
    epochHWV_T = zeros(6,n)
    epochHWV_h = zeros(6,n)

    #HWV.state.T.A1, HWV.state.T.A2, HWV.A1.state.T.A1, HWV.A1.state.T.A2, HWV.A2.state.T.A1, HWV.A2.state.T.A2 = 0.7, 0.7, 1.0, 1.0, 1.0, 1.0
    ## Param init conditions
    P𝑮 = ones(n+1)
    r𝑮 = 0.0
    h_avg = 0.0

    for i=1:n
        old_R = sum([HWV.state.R HWV.A1.state.R HWV.A2.state.R])
        if rand() < P𝑮[i]
            HWV = GDCtrl(HWV,α=α,θ=θ)
            r𝑮 = sum([HWV.state.R HWV.A1.state.R HWV.A2.state.R]) - old_R
        else
            HWV = habitCtrl(HWV,α=α,θ=θ)
            h_avg = sum([HWV.state.h.A1, HWV.state.h.A2, HWV.A1.state.h.A1, HWV.A1.state.h.A2, HWV.A2.state.h.A1, HWV.A2.state.h.A2])/6

        end
        epochHWV_Q[:,i] = [HWV.state.Q.A1, HWV.state.Q.A2, HWV.A1.state.Q.A1, HWV.A1.state.Q.A2, HWV.A2.state.Q.A1, HWV.A2.state.Q.A2]
        epochHWV_T[:,i] = [HWV.state.T.A1, HWV.state.T.A2, HWV.A1.state.T.A1, HWV.A1.state.T.A2, HWV.A2.state.T.A1, HWV.A2.state.T.A2]
        epochHWV_h[:,i] = [HWV.state.h.A1, HWV.state.h.A2, HWV.A1.state.h.A1, HWV.A1.state.h.A2, HWV.A2.state.h.A1, HWV.A2.state.h.A2]

        rwd_diff = (r𝑮 - sum([HWV.state.R HWV.A1.state.R HWV.A2.state.R])) #GD.state.R - (h.state.R + Gd.state.R)
        P𝑮[i+1] = 1/( 1 + exp(abs(rwd_diff) - abs(h_avg^2)) )
    end

    return HWV, epochHWV_Q, epochHWV_T, epochHWV_h, P𝑮

end

function runHabit(; habit::DecisionTree=buildStepTask(2), n::Int=1000, α::Float64=0.5, θ::Float64=5.0 )
    epochHabit = zeros(6,n)
    epochHabit_Q = zeros(6,n)
    for i=1:n
        habit = habitCtrl(habit,α=α,θ=θ)
        epochHabit[:,i] = [habit.state.h.A1, habit.state.h.A2, habit.A1.state.h.A1, habit.A1.state.h.A2, habit.A2.state.h.A1, habit.A2.state.h.A2]
        epochHabit_Q[:,i] = [habit.state.Q.A1, habit.state.Q.A2, habit.A1.state.Q.A1, habit.A1.state.Q.A2, habit.A2.state.Q.A1, habit.A2.state.Q.A2]
    end

    return habit,epochHabit_Q,Nothing(), epochHabit
end

function runMF(; MF::DecisionTree=buildStepTask(2), n::Int=1000, α::Float64=0.5, θ::Float64=5.0)
    epochMF = zeros(6,n)
    for i=1:n
        MF = MFCtrl(MF,α=α,θ=θ)
        epochMF[:,i] = [MF.state.Q.A1, MF.state.Q.A2, MF.A1.state.Q.A1, MF.A1.state.Q.A2, MF.A2.state.Q.A1, MF.A2.state.Q.A2]
    end

    return MF, epochMF, Nothing(), Nothing()
end

function runMB(; MB::DecisionTree=buildStepTask(2,TM=true), n::Int=1000, α::Float64=0.5, θ::Float64=5.0, TM::AbstractArray = [0.7 0.7 1.0 1.0 1.0 1.0] )
    MB.state.T.A1, MB.state.T.A2, MB.A1.state.T.A1, MB.A1.state.T.A2, MB.A2.state.T.A1, MB.A2.state.T.A2 = TM[1], TM[2], TM[3], TM[4], TM[5], TM[6]
    epochMB_Q = zeros(6,n)
    epochMB_T = zeros(6,n)
    for i=1:n
        MB = MBCtrl(MB,α=α,θ=θ)
        epochMB_Q[:,i] = [MB.state.Q.A1, MB.state.Q.A2, MB.A1.state.Q.A1, MB.A1.state.Q.A2, MB.A2.state.Q.A1, MB.A2.state.Q.A2]
        epochMB_T[:,i] = [MB.state.T.A1, MB.state.T.A2, MB.A1.state.T.A1, MB.A1.state.T.A2, MB.A2.state.T.A1, MB.A2.state.T.A2]
    end

    return MB, epochMB_Q, epochMB_T, Nothing()
end

function runGD(; GD::DecisionTree=buildStepTask(2,TM=true), n::Int=1000, α::Float64=0.5, θ::Float64=5.0 )
    epochGD_Q = zeros(6,n)
    epochGD_T = zeros(6,n)
    for i=1:n
        GD = GDCtrl(GD,α=α,αₜ=0.05,θ=θ)
        epochGD_Q[:,i] = [GD.state.Q.A1, GD.state.Q.A2, GD.A1.state.Q.A1, GD.A1.state.Q.A2, GD.A2.state.Q.A1, GD.A2.state.Q.A2]
        epochGD_T[:,i] = [GD.state.T.A1, GD.state.T.A2, GD.A1.state.T.A1, GD.A1.state.T.A2, GD.A2.state.T.A1, GD.A2.state.T.A2]
    end

    return GD, epochGD_Q, epochGD_T,Nothing()
end

function runAll(; n::Int=1000, α::Float64=0.5, θ::Float64=5.0 )
    ## Raw Models
    habit = runHabit(N=n,α=α,θ=θ)
    MF = runMF(N=n,α=α,θ=θ)
    MB = runMB(N=n,α=α,θ=θ)
    # Blend/variation Models
    GD = runGD(N=n,α=α,θ=θ)
    HWV = runHWV(N=n,α=α,θ=θ)

    return habit, MF, MB, GD, HWV
end

function plotAll(; n::Int=1000, α::Float64=0.5, θ::Float64=5.0 )
    ## Raw Models
    habit = plotSim(runHabit,N=n,α=α,θ=θ)
    MF = plotSim(runMF,N=n,α=α,θ=θ)
    MB = plotSim(runMB,N=n,α=α,θ=θ)
    # Blend/variation Models
    GD = plotSim(runGD,N=n,α=α,θ=θ)
    HWV = plotSim(runHWV,N=n,α=α,θ=θ)

    return habit, MF, MB, GD, HWV

end

function plotSim(f::Function; N::Int=1000, α::Float64=0.5, θ::Float64=5.0 )

    Model = f(n=N,α=α,θ=θ)
    plt_Q,bar_Q,plt_T,bar_T,plt_h,bar_h,plt_arb = Nothing(),Nothing(),Nothing(),Nothing(),Nothing(),Nothing(),Nothing()
    m = "$f"[4:end]

    if Model[2] != nothing

        anaQ = zeros(8,N)
        for i=1:N-1
            anaQ[1,i+1] = (1-α)*anaQ[1,i] + α*(0.7*anaQ[3,i]+0.3*anaQ[4,i])
            anaQ[2,i+1] = (1-α)*anaQ[2,i] + α*(0.3*anaQ[3,i]+0.7*anaQ[4,i])
            anaQ[3,i+1] = (1-α)*anaQ[3,i] + α*(findmax([anaQ[5,i] anaQ[6,i]])[1])
            anaQ[4,i+1] = (1-α)*anaQ[4,i] + α*(findmax([anaQ[7,i] anaQ[8,i]])[1])
            anaQ[5,i+1] = (1-α)*anaQ[5,i] + α*Model[1].A1.A1
            anaQ[6,i+1] = (1-α)*anaQ[6,i] + α*Model[1].A1.A2
            anaQ[7,i+1] = (1-α)*anaQ[7,i] + α*Model[1].A2.A1
            anaQ[8,i+1] = (1-α)*anaQ[8,i] + α*Model[1].A2.A2
        end

        plt_Q = plot(Model[2][1,:],label="A1",ylims = (0, 1))
        plot!(Model[2][2,:],label="A2")
        plot!(Model[2][3,:],label="A1.A1")
        plot!(Model[2][4,:],label="A1.A2")
        plot!(Model[2][5,:],label="A2.A1")
        plot!(Model[2][6,:],label="A2.A2")
        plot!(anaQ[1,:],label="Analytic A1",color="blue",linestyle=:dash)
        plot!(anaQ[2,:],label="Analytic A2",color="orange",linestyle=:dash)
        plot!(anaQ[3,:],label="Analytic A1.A1",color="green",linestyle=:dash)
        plot!(anaQ[4,:],label="Analytic A2.A1",color="magenta",linestyle=:dash)

        title!("$m Time Series of Q values")
        xaxis!("Number of iterations")
        yaxis!("Q(s,a)")

        bar_Q = bar([Model[1].state.Q.A1, Model[1].state.Q.A2, Model[1].A1.state.Q.A1, Model[1].A1.state.Q.A2, Model[1].A2.state.Q.A1, Model[1].A2.state.Q.A2],legend=false,ylims = (0, 1));
        title!("$m Final Q values")
        xaxis!("Action(s) Taken")
        xticks!((1:6),["A1", "A2", "A1.A1", "A1.A2", "A2.A1", "A2.A2"])
        yaxis!("Q(s,a)")
    end

    if Model[1].state.T != nothing

        plt_T = plot(Model[3][1,:],label="A1",ylims = (0, 1))
        plot!(Model[3][2,:],label="A2")
        plot!(Model[3][3,:],label="A1.A1")
        plot!(Model[3][4,:],label="A1.A2")
        plot!(Model[3][5,:],label="A2.A1")
        plot!(Model[3][6,:],label="A2.A2")
        title!("$m Time Series of Transition Model")

        xaxis!("Number of iterations")
        yaxis!("T(s,a,s')")

        bar_T = bar([Model[1].state.T.A1, Model[1].state.T.A2, Model[1].A1.state.T.A1, Model[1].A1.state.T.A2, Model[1].A2.state.T.A1, Model[1].A2.state.T.A2],legend=false,ylims = (0, 1));
        title!("$m Final Transition Model")
        xaxis!("Action(s) Taken")
        xticks!((1:6),["A1", "A2", "A1.A1", "A1.A2", "A2.A1", "A2.A2"])
        yaxis!("T(s,a,s')")
    end

    if Model[4] != nothing
        plt_h = plot(Model[4][1,:],label="A1",ylims = (0, 1))
        plot!(Model[4][2,:],label="A2")
        plot!(Model[4][3,:],label="A1.A1")
        plot!(Model[4][4,:],label="A1.A2")
        plot!(Model[4][5,:],label="A2.A1")
        plot!(Model[4][6,:],label="A2.A2")

        title!("$m Time Series of values")
        xaxis!("Number of iterations")
        yaxis!("h(s,a)")

        bar_h = bar([Model[1].state.h.A1, Model[1].state.h.A2, Model[1].A1.state.h.A1, Model[1].A1.state.h.A2, Model[1].A2.state.h.A1, Model[1].A2.state.h.A2],legend=false,ylims = (0, 1));
        title!("$m Final Habit values")
        xaxis!("Action(s) Taken")
        xticks!((1:6),["A1", "A2", "A1.A1", "A1.A2", "A2.A1", "A2.A2"])
        yaxis!("h(s,a)")
    end
    if Model[1].state.T != nothing && Model[4] != nothing
        plt_arb = plot(Model[5],ylims = (0, 1))
        title!("Probability over time of Goal Directed Controller being chosen")
        xaxis!("Number of iterations")
        yaxis!("Probability of Goal Directed Controller being chosen")
    end

    return Model,plt_Q,bar_Q,plt_T,bar_T,plt_h,bar_h,plt_arb
end

habitSimResults = plotSim(runHabit,N=5000)
MFSimResults = plotSim(runMF,N=5000)
MBSimResults = plotSim(runMB,N=5000)
GDSimResults = plotSim(runGD,N=5000)
HWVSimResults = plotSim(runHWV,N=5000,α=0.5)

habitSimResults[1][1]

plth = habitSimResults[4]
pltf = MFSimResults[1]
pltb = MBSimResults[1]
pltg = GDSimResults[1]

plot(HWVSimResults[1],HWVSimResults[3],HWVSimResults[5],HWVSimResults[7],legend=false)
