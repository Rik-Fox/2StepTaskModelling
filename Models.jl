using Random, Plots
pyplot()
push!(LOAD_PATH, pwd())
using CustomStructs

function buildStepTask(steps;TM=false,branch=Nothing(),r=0.0)

    if steps == 1

        Q = Actions(0.0,0.0)
        if TM == true
            T = Actions(0.0,0.0)
        else
            T = Nothing()
        end
        R = 0.0

        Task = DecisionTree(State(Q,T,R),round(0.7-r,digits=2),round(0.3-r,digits=2))

    else

        Q = Actions(0.0,0.0)
        if TM == true
            T = Actions(0.0,0.0)
        else
            T = Nothing()
        end
        R = 0.0

        Task = DecisionTree(State(Q,T,R), buildStepTask(steps-1,TM=TM,branch=1,r=0.0), buildStepTask(steps-1,TM=TM,branch=0,r=0.2))
    end

    return Task
end

function softMax(a, A; θ::Float64=1.0)
    p = exp(θ*a)/sum(exp.(θ*A))
    return p
end

function rwd(p)

    if rand() < p
        return 10.0
    else
        return 0.0
    end

end

function transitionUpdate(state::String,T::Actions,α::Float64)
    if state == "A1"
        T.A1 = (1-α)*T.A1 + α
        T.A2 = (1-α)*T.A2
    elseif state == "A2"
        T.A1 = (1-α)*T.A1
        T.A2 = (1-α)*T.A2 + α
    else
        throw(ArgumentError("First argument must be either \"A1\" or \"A2\""))
    end
    return T
end

function modelledQUpdate(Node::DecisionTree, path::String, r::Float64 α::Float64; leaf::Bool)

    if  leaf == true
        if path == "A1"
            Node.state.Q.A1 = (1-α)*Node.state.Q.A1 + α*r
        elseif path == "A2"
            Node.state.Q.A2 = (1-α)*Node.state.Q.A2 + α*r
        else
            throw(ArgumentError("path argument must be either \"A1\" or \"A2\""))
        end
    else
        Q_ = Node.state.T.A1*(Node.A1.state.Q.A2*Node.A1.state.T.A2 + Node.A1.state.Q.A1*Node.A1.state.T.A1) + Node.state.T.A2*(Node.A2.state.Q.A2*Node.A2.state.T.A2 + Node.A2.state.Q.A1*Node.A2.state.T.A1)

        if path == "A1"
            Node.state.Q.A2 = (1-α)*Node.state.Q.A2 + α*Q_
        elseif path == "A2"
            Node.state.Q.A2 = (1-α)*Node.state.Q.A2 + α*Q_
        else
            throw(ArgumentError("path argument must be either \"A1\" or \"A2\""))
        end
    end
end

function switchPath(L::Bool, α::Float64, Node::DecisionTree, switchNode::DecisionTree, something::Function)

    if L == true
        A1, A2 = deepcopy(Node.A1), deepcopy(Node.A2)
        Node.A1, Node.A2 = deepcopy(switchNode.A1), deepcopy(switchNode.A2)
        Node = GDCtrl(Node,α=α)
        Node.A1, Node.A2 = A1, A2
    else
        Node = GDCtrl(Node,α=α)
    end

    return Node
end

function Qlearn(Node::DecisionTree; α::Float64=1.0, branch=Nothing())

    A = [Node.state.Q.A1, Node.state.Q.A1]

    a_idx = findall(x->x==findmax(A)[1], A)

    π = softMax(A[a_idx][1], A)

    rv = rand()

    if typeof(Node.A1) == Nothing || typeof(Node.A2) == Nothing

        if π > rv && a_idx == 1 || π < rv && a_idx == 2
            Node.state.Q.A1 = (1-α)*Node.state.Q.A1 + α*Node.state.R
        elseif π > rv && a_idx == 2 || π < rv && a_idx == 1
            Node.state.Q.A2 = (1-α)*Node.state.Q.A2 + α*Node.state.R
        end

    else
        if π > rv && a_idx == 1 || π < rv && a_idx == 2
            Node.A1 = Qlearn(Node.A1,branch=1,α=α)
        elseif π > rv && a_idx == 2 || π < rv && a_idx == 1
            Node.A2 = Qlearn(Node.A2,branch=0,α=α)
        end
        Q_ = findmax(A)
        Node.state.Q = (1-α)*Node.state.Q + α*Q_
    end

    return Node

end

function habitCtrl(Node::DecisionTree; α::Float64=0.0, branch=Nothing())

    A = [Node.state.Q.A1, Node.state.Q.A2]

    a_idx = findall(x->x==findmax(A)[1], A)[1]

    π = softMax(A[a_idx][1], A)

    rv = rand()

    if typeof(Node.A1) == Float64 || typeof(Node.A2) == Float64

        if π > rv && a_idx == 1 || π < rv && a_idx == 2
            Node.state.Q.A1 = (1-α)*Node.state.Q.A1 + α
            Node.state.Q.A2 = (1-α)*Node.state.Q.A2
        elseif π > rv && a_idx == 2 || π < rv && a_idx == 1
            Node.state.Q.A2 = (1-α)*Node.state.Q.A2 + α
            Node.state.Q.A1 = (1-α)*Node.state.Q.A1
        end
    else
        if π > rv && a_idx == 1 || π < rv && a_idx == 2
            Node.A1 = habitCtrl(Node.A1,branch=1,α=α)
            Node.state.Q.A1 = (1-α)*Node.state.Q.A1 + α
            Node.state.Q.A2 = (1-α)*Node.state.Q.A2

        elseif π > rv && a_idx == 2 || π < rv && a_idx == 1
            Node.A2 = habitCtrl(Node.A2,branch=0,α=α)
            Node.state.Q.A2 = (1-α)*Node.state.Q.A2 + α
            Node.state.Q.A1 = (1-α)*Node.state.Q.A1
        end

    end

    return Node

end

function MFCtrl(Node::DecisionTree; α::Float64=1.0, branch=Nothing())

    A = [Node.state.Q.A1, Node.state.Q.A1]
    a_idx = findall(x->x==findmax(A)[1], A)[1]
    π = softMax(A[a_idx][1], A)

    rv = rand()

    if typeof(Node.A1) == Float64 || typeof(Node.A2) == Float64
        if ((π > rv && a_idx == 1) || (π < rv && a_idx == 2))
            r=Node.A1
            path = "A1"
        else
            r=Node.A2
            path = "A2"
        end
        QUpdate(Node,path,rwd(r),α,leaf=true)
    else
        if (π > rv && a_idx == 1) || (π < rv && a_idx == 2)
            if rand() > 0.7
                Node.A1 = MFCtrl(Node.A2,α=α)
            else
                Node.A1 = MFCtrl(Node.A1,α=α)
            end
            Q_ = findmax([Node.A1.state.Q.A1, Node.A1.state.Q.A2])[1]
            Node.state.Q.A1 = (1-α)*Node.state.Q.A1 + α*Q_
        elseif (π > rv && a_idx == 2) || (π < rv && a_idx == 1)
            if rand() > 0.7
                Node.A2 = MFCtrl(Node.A1,α=α)
            else
                Node.A2 = MFCtrl(Node.A2,α=α)
            end
            Q_ = findmax([Node.A2.state.Q.A1, Node.A2.state.Q.A2])[1]
            Node.state.Q.A2 = (1-α)*Node.state.Q.A2 + α*Q_
        end
    end

    return Node
end

function MBCtrl(Node::DecisionTree; α::Float64=1.0, branch=Nothing())

    A = [Node.state.Q.A1, Node.state.Q.A1]
    a_idx = findall(x->x==findmax(A)[1], A)[1]
    π = softMax(A[a_idx][1], A)

    rv = rand()

    if typeof(Node.A1) == Float64 || typeof(Node.A2) == Float64
        if ((π > rv && a_idx == 1) || (π < rv && a_idx == 2))
            r=Node.A1
            path = "A1"
        else
            r=Node.A2
            path = "A2"
        end
        modelledQUpdate(Node,path,rwd(r),α,leaf=true)
    else
        if (π > rv && a_idx == 1) || (π < rv && a_idx == 2)
            path = "A1"
            switch=false
            if rand() > 0.7; switch=true; path = "A2"; end
            switchPath(switch,α,Node.A1,Node.A2)
            modelledQUpdate(Node,path,0.0,α,leaf=false)
        elseif (π > rv && a_idx == 2) || (π < rv && a_idx == 1)
            path = "A2"
            switch=false
            if rand() > 0.7; switch=true; path = "A1"; end
            switchPath(switch,α,Node.A2,Node.A1)
            modelledQUpdate(Node,path,0.0,α,leaf=false)
        end
    end

    return Node
end

function GDCtrl(Node::DecisionTree; α::Float64=1.0, branch=Nothing())
    A = [Node.state.Q.A1, Node.state.Q.A1]
    a_idx = findall(x->x==findmax(A)[1], A)[1]
    π = softMax(A[a_idx][1], A)

    rv = rand()

    if typeof(Node.A1) == Float64 || typeof(Node.A2) == Float64
        if ((π > rv && a_idx == 1) || (π < rv && a_idx == 2))
            r=Node.A1
            path = "A1"
        else
            r=Node.A2
            path = "A2"
        end
        transitionUpdate(path,Node.state.T,α)
        modelledQUpdate(Node,path,rwd(Node.A1),α,leaf=true)
    else
        if (π > rv && a_idx == 1) || (π < rv && a_idx == 2)
            path = "A1"
            switch=false
            if rand() > 0.7; switch=true; path = "A2"; end
            switchPath(switch,α,Node.A1,Node.A2)
            transitionUpdate(path,Node.state.T,α)
            modelledQUpdate(Node,path,0.0,α,leaf=false)
        elseif (π > rv && a_idx == 2) || (π < rv && a_idx == 1)
            path = "A2"
            switch=false
            if rand() > 0.7; switch=true; path = "A1"; end
            switchPath(switch,α,Node.A2,Node.A1)
            transitionUpdate(path,Node.state.T,α)
            modelledQUpdate(Node,path,0.0,α,leaf=false)
        end
    end

    return Node
end

function plotQT(Node)
    plt1 = bar([Node.state.Q.A1, Node.state.Q.A2, Node.A1.state.Q.A1, Node.A1.state.Q.A2, Node.A2.state.Q.A1, Node.A2.state.Q.A2]);
    plt2 = bar([Node.state.T.A1, Node.state.T.A2, Node.A1.state.T.A1, Node.A1.state.T.A2, Node.A2.state.T.A1, Node.A2.state.T.A2]);

    return plot(plt1,plt2,layout=(2,1))

end

function runHabit(;habit::DecisionTree=buildStepTask(2),η=1.0,n::Int=100)
    for i=1:n
        η *= 1/(1+(0.5*i))
        habit = habitCtrl(habit,α=η)
    end

    pltHabit = bar([habit.state.Q.A1, habit.state.Q.A2, habit.A1.state.Q.A1, habit.A1.state.Q.A2, habit.A2.state.Q.A1, habit.A2.state.Q.A2])
    return habit, pltHabit
end


function runMF(;MF::DecisionTree=buildStepTask(2),η=1.0,n::Int=100)
    for i=1:n
        η *= 1/(1+(0.5*i))
        MF = MFCtrl(MF,α=η)
    end

    pltMB = bar([MF.state.Q.A1, MF.state.Q.A2, MF.A1.state.Q.A1, MF.A1.state.Q.A2, MF.A2.state.Q.A1, MF.A2.state.Q.A2])
    return MB, pltMB
end


function runMB(;MB::DecisionTree=buildStepTask(2,TM=true),η=1.0,n::Int=100)
    for i=1:n
        η *= 1/(1+(0.5*i))
        MB = MBCtrl(MB,α=η)
    end

    pltMB = plotQT(MB)
    return MB, pltMB
end

function runGD(;GD::DecisionTree=buildStepTask(2,TM=true),η=1.0,n::Int=100)
    for i=1:n
        η *= 1/(1+(0.5*i))
        GD = GDCtrl(GD,α=η)
    end
    pltGD = plotQT(GD)

    return GD, pltGD
end

function runModels()
    runHabit()
    runMF()
    runMB()
    runGD()
end

runModels()
