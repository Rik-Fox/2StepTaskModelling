using Random

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
