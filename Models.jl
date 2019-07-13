using Random, Plots
pyplot()
push!(LOAD_PATH, pwd())
using CustomStructs

function buildStepTask(steps::Int;TM::Bool=false,branch=Nothing(),r::Float64=0.0)
    if steps == 1
        Q = Actions(0.0,0.0)
        TM ? T = Actions(0.5,0.5) : T = Nothing()
        R = 0.0
        Task = DecisionTree(State(Q,T,R),round(0.8-r,digits=2),0.0)

    else
        Q = Actions(0.0,0.0)
        TM ? T = Actions(0.5,0.5) : T = Nothing()
        R = 0.0
        Task = DecisionTree(State(Q,T,R), buildStepTask(steps-1,TM=TM), buildStepTask(steps-1,TM=TM,r=0.6))
    end

    return Task
end

function softMax(a, A; θ::Float64=1.0)
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

function habitUpdate(state::String,h::Actions,α::Float64)
    if state == "A1"
        h.A1 = (1-α)*h.A1 + α
        h.A2 = (1-α)*h.A2
    elseif state == "A2"
        h.A1 = (1-α)*h.A1
        h.A2 = (1-α)*h.A2 + α
    else
        throw(ArgumentError("First argument must be either \"A1\" or \"A2\""))
    end
    return h
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
        elseif actn == "A2"
            Node.state.Q.A2 = (1-α)*Node.state.Q.A2 + α*r
        else
            throw(ArgumentError("action argument must be either \"A1\" or \"A2\""))
        end
    else

        actn == "A1" ? (p = Node.state.T.A1 ; q = 1-(Node.state.T.A1)) : (p = 1-(Node.state.T.A2) ; q = Node.state.T.A2)

        Q_ = p*findmax([Node.A1.state.Q.A2  Node.A1.state.Q.A1])[1] + q*findmax([Node.A2.state.Q.A2  Node.A2.state.Q.A1])[1]

        if actn == "A1"
            Node.state.Q.A1 = (1-α)*Node.state.Q.A1 + α*Q_
        elseif actn == "A2"
            Node.state.Q.A2 = (1-α)*Node.state.Q.A2 + α*Q_
        else
            throw(ArgumentError("action argument must be either \"A1\" or \"A2\""))
        end
    end
end

function QUpdate(Node::DecisionTree, actn::String, r::Float64, α::Float64; leaf::Bool=false)

    if  leaf == true
        if actn == "A1"
            Node.state.Q.A1 = (1-α)*Node.state.Q.A1 + α*r
        elseif actn == "A2"
            Node.state.Q.A2 = (1-α)*Node.state.Q.A2 + α*r
        else
            throw(ArgumentError("action argument must be either \"A1\" or \"A2\""))
        end
    else

        if actn == "A1"
            Q_ = findmax([Node.A1.state.Q.A1, Node.A1.state.Q.A2])[1]
            Node.state.Q.A1 = (1-α)*Node.state.Q.A1 + α*Q_
        elseif actn == "A2"
            Q_ = findmax([Node.A2.state.Q.A1, Node.A2.state.Q.A2])[1]
            Node.state.Q.A2 = (1-α)*Node.state.Q.A2 + α*Q_
        else
            throw(ArgumentError("action argument must be either \"A1\" or \"A2\""))
        end
    end

    return Node
end

function habitCtrl(Node::DecisionTree; α::Float64=0.5)

    A = [Node.state.Q.A1, Node.state.Q.A2]
    a_idx = findall(x->x==findmax(A)[1], A)[1]
    π = softMax(A[a_idx][1], A)

    rv = rand()

    if typeof(Node.A1) == Float64 || typeof(Node.A2) == Float64
        ((π >= rv && a_idx == 1) || (π < rv && a_idx == 2)) ? habitUpdate("A1",Node.state.Q,α) : habitUpdate("A2",Node.state.Q,α)
    else
        if π >= rv && a_idx == 1 || π < rv && a_idx == 2
            rand() < 0.7 ? (sw=false;actn = "A1") : (sw=true;actn = "A2")
            switchActn(sw,α,Node.A1,Node.A2,habitCtrl)
            habitUpdate("A1",Node.state.Q,α)
        elseif π >= rv && a_idx == 2 || π < rv && a_idx == 1
            rand() < 0.7 ? (sw=false;actn = "A2") : (sw=true;actn = "A1")
            switchActn(sw,α,Node.A2,Node.A1,habitCtrl)
            habitUpdate("A2",Node.state.Q,α)
        end
    end

    return Node
end

function MFCtrl(Node::DecisionTree; α::Float64=0.5)

    A = [Node.state.Q.A1, Node.state.Q.A2]
    a_idx = findall(x->x==findmax(A)[1], A)[1]
    π = softMax(A[a_idx][1], A)

    rv = rand()

    if typeof(Node.A1) == Float64 || typeof(Node.A2) == Float64
        ((π >= rv && a_idx == 1) || (π < rv && a_idx == 2)) ? (r=Node.A1;actn = "A1") : (r=Node.A2;actn = "A2")
        QUpdate(Node,actn,rwd(r),α,leaf=true)
    else
        if (π >= rv && a_idx == 1) || (π < rv && a_idx == 2)
            actn = "A1"
            rand() < 0.7 ? sw=false : sw=true
            switchActn(sw,α,Node.A1,Node.A2,MFCtrl)
            QUpdate(Node,"A1",0.0,α)
        elseif (π >= rv && a_idx == 2) || (π < rv && a_idx == 1)
            actn = "A2"
            rand() < 0.7 ? sw=false : sw=true
            switchActn(sw,α,Node.A2,Node.A1,MFCtrl)
            QUpdate(Node,"A2",0.0,α)
        end
    end

    return Node
end

function MBCtrl(Node::DecisionTree; α::Float64=0.5)
    A = [Node.state.Q.A1, Node.state.Q.A2]
    a_idx = findall(x->x==findmax(A)[1], A)[1]
    π = softMax(A[a_idx][1], A)

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

function GDCtrl(Node::DecisionTree; α::Float64=0.5, αₜ::Float64=0.1)
    A = [Node.state.Q.A1, Node.state.Q.A2]
    a_idx = findall(x->x==findmax(A)[1], A)[1]
    π = softMax(A[a_idx][1], A)

    rv = rand()

    if typeof(Node.A1) == Float64 || typeof(Node.A2) == Float64
        ((π >= rv && a_idx == 1) || (π < rv && a_idx == 2)) ? (r=Node.A1;actn = "A1") : (r=Node.A2;actn = "A2")
        transitionUpdate(actn,Node.state.T,αₜ)
        modelledQUpdate(Node,actn,rwd(r),α,leaf=true)
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

function compositeModel()

    HWV = buildStepTask(2,TM=true)

    h = habitCtrl(h,α=)
    GD = GDCtrl(GD,α=)

    arb = 1/(1 + exp( W𝑮*(r𝐺-r₀)^2 - Wₕ*mean(h^2)))
end

function plotQT(Node)
    plt1 = bar([Node.state.Q.A1, Node.state.Q.A2, Node.A1.state.Q.A1, Node.A1.state.Q.A2, Node.A2.state.Q.A1, Node.A2.state.Q.A2]);
    plt2 = bar([Node.state.T.A1, Node.state.T.A2, Node.A1.state.T.A1, Node.A1.state.T.A2, Node.A2.state.T.A1, Node.A2.state.T.A2]);

    return plot(plt1,plt2,layout=(2,1))
end

function runHabit(;habit::DecisionTree=buildStepTask(2),n::Int=1000)
    epochHabit = zeros(6,n)
    for i=1:n
        habit = habitCtrl(habit,α=0.005)
        epochHabit[:,i] = [habit.state.Q.A1, habit.state.Q.A2, habit.A1.state.Q.A1, habit.A1.state.Q.A2, habit.A2.state.Q.A1, habit.A2.state.Q.A2]
    end
    pltHabit = bar([habit.state.Q.A1, habit.state.Q.A2, habit.A1.state.Q.A1, habit.A1.state.Q.A2, habit.A2.state.Q.A1, habit.A2.state.Q.A2])

    return habit, pltHabit, epochHabit
end

function runMF(;MF::DecisionTree=buildStepTask(2),n::Int=1000)
    epochMF = zeros(6,n)
    for i=1:n
        MF = MFCtrl(MF,α=0.01)
        epochMF[:,i] = [MF.state.Q.A1, MF.state.Q.A2, MF.A1.state.Q.A1, MF.A1.state.Q.A2, MF.A2.state.Q.A1, MF.A2.state.Q.A2]
    end
    pltMF = bar([MF.state.Q.A1, MF.state.Q.A2, MF.A1.state.Q.A1, MF.A1.state.Q.A2, MF.A2.state.Q.A1, MF.A2.state.Q.A2])

    return MF, pltMF, epochMF
end

function runMB(;MB::DecisionTree=buildStepTask(2,TM=true),n::Int=1000, TM::AbstractArray = [0.7 0.7 1.0 1.0 1.0 1.0])
    MB.state.T.A1, MB.state.T.A2, MB.A1.state.T.A1, MB.A1.state.T.A2, MB.A2.state.T.A1, MB.A2.state.T.A2 = TM[1], TM[2], TM[3], TM[4], TM[5], TM[6]
    epochMB_Q = zeros(6,n)
    epochMB_T = zeros(6,n)
    for i=1:n
        MB = MBCtrl(MB,α=0.01)
        epochMB_Q[:,i] = [MB.state.Q.A1, MB.state.Q.A2, MB.A1.state.Q.A1, MB.A1.state.Q.A2, MB.A2.state.Q.A1, MB.A2.state.Q.A2]
        epochMB_T[:,i] = [MB.state.T.A1, MB.state.T.A2, MB.A1.state.T.A1, MB.A1.state.T.A2, MB.A2.state.T.A1, MB.A2.state.T.A2]
    end
    pltMB = plotQT(MB)

    return MB, pltMB, epochMB_Q, epochMB_T
end

function runGD(;GD::DecisionTree=buildStepTask(2,TM=true),n::Int=1000)
    epochGD_Q = zeros(6,n)
    epochGD_T = zeros(6,n)
    for i=1:n
        GD = GDCtrl(GD,α=0.01,αₜ=0.01)
        epochGD_Q[:,i] = [GD.state.Q.A1, GD.state.Q.A2, GD.A1.state.Q.A1, GD.A1.state.Q.A2, GD.A2.state.Q.A1, GD.A2.state.Q.A2]
        epochGD_T[:,i] = [GD.state.T.A1, GD.state.T.A2, GD.A1.state.T.A1, GD.A1.state.T.A2, GD.A2.state.T.A1, GD.A2.state.T.A2]
    end

    pltGD = plotQT(GD)

    return GD, pltGD, epochGD_Q, epochGD_T
end

function runModels()
    #habit = runHabit()
    MF = runMF()
    MB = runMB()
    GD = runGD()

    return MF, MB, GD
end

function plotSim(f::Function;N::Int=1000,α=0.01)

    Model = f(n=N)
    pltT = Nothing()

    anaQ = zeros(10,N)
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

    if Model[1].state.T != nothing
        pltT = plot(Model[4][1,:],label="A1")
        plot!(Model[4][2,:],label="A2")
        plot!(Model[4][3,:],label="A1.A1")
        plot!(Model[4][4,:],label="A1.A2")
        plot!(Model[4][5,:],label="A2.A1")
        plot!(Model[4][6,:],label="A2.A2")
        xaxis!("Number of iterations")
        yaxis!("T(s,a,s')")

        plt = plot(Model[3][1,:],label="A1")
        plot!(Model[3][2,:],label="A2")
        plot!(Model[3][3,:],label="A1.A1")
        plot!(Model[3][4,:],label="A1.A2")
        plot!(Model[3][5,:],label="A2.A1")
        plot!(Model[3][6,:],label="A2.A2")

        plot!(anaQ[3,:],label="Analytic",color="green",linestyle=:dash)
        plot!(anaQ[1,:],label="Analytic",color="blue",linestyle=:dash)
        plot!(anaQ[2,:],label="Analytic",color="red",linestyle=:dash)
        plot!(anaQ[4,:],label="Analytic",color="tan",linestyle=:dash)

        xaxis!("Number of iterations")
        yaxis!("Q(s,a)")
    else

        plt = plot(Model[3][1,:],label="A1")
        plot!(Model[3][2,:],label="A2")
        plot!(Model[3][3,:],label="A1.A1")
        plot!(Model[3][4,:],label="A1.A2")
        plot!(Model[3][5,:],label="A2.A1")
        plot!(Model[3][6,:],label="A2.A2")

        plot!(anaQ[3,:],label="Analytic",color="green",linestyle=:dash)
        plot!(anaQ[1,:],label="Analytic",color="blue",linestyle=:dash)
        plot!(anaQ[2,:],label="Analytic",color="red",linestyle=:dash)
        plot!(anaQ[4,:],label="Analytic",color="tan",linestyle=:dash)
        xaxis!("Number of iterations")
        yaxis!("Q(s,a)")
    end

    return plt,pltT,Model[2]
end

habitSimResults = plotSim(runHabit,N=5000)
plot(habitSimResults[1],habitSimResults[3],layout=(2,1))

MFSimResults = plotSim(runMF,N=10000)
plot(MFSimResults[1])
plot(MFSimResults[1],MFSimResults[3],layout=(2,1))

MBSimResults = plotSim(runMB,N=10000)
plot(MBSimResults[1])
plot(MFSimResults[1],MBSimResults[2],MBSimResults[3],layout=3)

GDSimResults = plotSim(runGD,N=5000)
plot(GDSimResults[3],GDSimResults[2],GDSimResults[1],layout=3)

plot(habitSimResults[1],MFSimResults[1],MFSimResults[1],GDSimResults[1],layout=(2,2))
