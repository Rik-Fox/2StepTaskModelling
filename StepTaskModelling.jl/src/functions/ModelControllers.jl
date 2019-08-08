SCimport Random

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

    π = softMax([Node.state.h.A1, Node.state.h.A2],θ=θ)

    if typeof(data) == Nothing
        π >= rand() ? actn = "A1" : actn = "A2"

        μ, Rwd = taskEval(Node.state.name,actn)
        Node.state.R = Rwd
        Node.state.h = habitUpdate(Node.state.h,actn,α)
        if Node.state.name == "ξ"
            μ ? SC=habitCtrl(Node.μ,α=α,θ=θ) : SC=MFCtrl(Node.ν,α=α,θ=θ)
            Rwd = SC[2]
            actn = actn, μ, SC[3], SC[4]
        end

    else
        data[1] ? actn = "A1" : actn = "A2"

        μ, Rwd = taskRead(Node.state.name,data[1:2])
        Node.state.R = Rwd
        Node.state.h = habitUpdate(Node.state.h,actn,α)
        if Node.state.name == "ξ"
            μ ? SC=habitCtrl(Node.μ,data=data[3:4],α=α,θ=θ) : SC=habitCtrl(Node.ν,data=data[3:4],α=α,θ=θ)
            Rwd = SC[2]
            actn = actn, μ, SC[3], SC[4]
        end
    end

    return Node, Rwd, actn, π
end

function MFCtrl(Node::DecisionTree; data::Union{AbstractArray,Nothing}=Nothing(), α::Float64=0.5,θ::Float64=5.0)

    π = softMax([Node.state.Q.A1, Node.state.Q.A2],θ=θ)

    if typeof(data) == Nothing
        π >= rand() ? actn = "A1" : actn = "A2"
        μ, Rwd = taskEval(Node.state.name,actn)
        Node.state.R = Rwd
        Node.state.Q = QUpdate(Node,actn,μ,α)
        if Node.state.name == "ξ"
            μ ? SC=MFCtrl(Node.μ,α=α,θ=θ) : SC=MFCtrl(Node.ν,α=α,θ=θ)
            Rwd = SC[2]
            actn = actn, μ, SC[3], SC[4]
        end
    else
        data[1] ? actn = "A1" : actn = "A2"

        μ, Rwd = taskRead(Node.state.name,data[1:2])
        Node.state.R = Rwd
        Node.state.Q = QUpdate(Node,actn,μ,α)
        if Node.state.name == "ξ"
            μ ? SC=MFCtrl(Node.μ,data=data[3:4],α=α,θ=θ) : SC=MFCtrl(Node.ν,data=data[3:4],α=α,θ=θ)
            Rwd = SC[2]
            actn = actn, μ, SC[3], SC[4]
        end
    end

    return Node, Rwd, actn, π
end

function MBCtrl(Node::DecisionTree; data::Union{AbstractArray,Nothing}=Nothing(), α::Float64=0.5,θ::Float64=5.0)
    if typeof(Node.state.T) == Nothing
        throw(error("Agent must have Transition modelling enabled, set kwarg \"TM=true\" when created"))
    end
    π = softMax([Node.state.Q.A1, Node.state.Q.A2],θ=θ)
    Node.state.Q.A1>Node.state.Q.A2 ? dat = "A1" : dat = "A2"
    if typeof(data) == Nothing
        π >= rand() ? actn = "A1" : actn = "A2"

        μ, Rwd = taskEval(Node.state.name,actn)
        Node.state.R = Rwd
        Node.state.Q = modelledQUpdate(Node,actn,μ,α)

        if Node.state.name == "ξ"
            μ ? SC=MBCtrl(Node.μ,α=α,θ=θ) : SC=MBCtrl(Node.ν,α=α,θ=θ)
            Rwd = SC[2]
            actn = actn, μ, SC[3], SC[4]
        end
    else
        data[1] ? actn = "A1" : actn = "A2"

        μ, Rwd = taskRead(Node.state.name,data[1:2])
        Node.state.R = Rwd
        Node.state.Q = modelledQUpdate(Node,actn,μ,α)

        if Node.state.name == "ξ"
            μ ? SC=MBCtrl(Node.μ,data=data[3:4],α=α,θ=θ) : SC=MBCtrl(Node.ν,data=data[3:4],α=α,θ=θ)
            Rwd = SC[2]
            actn = actn, μ, SC[3], SC[4]
        end
    end


    return Node, Rwd, actn, π
end

function GDCtrl(Node::DecisionTree; data::Union{AbstractArray,Nothing}=Nothing(), α::Float64=0.5, αₜ::Float64=0.1,θ::Float64=5.0)
    if typeof(Node.state.T) == Nothing
        throw(error("Agent must have Transition modelling enabled, set kwarg \"TM=true\" when created"))
    end

    π = softMax([Node.state.Q.A1, Node.state.Q.A2],θ=θ)

    if typeof(data) == Nothing
        π >= rand() ? actn = "A1" : actn = "A2"

        μ, Rwd = taskEval(Node.state.name,actn)
        Node.state.R = Rwd
        Node.state.Q = modelledQUpdate(Node,actn,μ,α)
        Node.state.T = transitionUpdate(Node,actn,μ,αₜ)

        if Node.state.name == "ξ"
            μ ? SC=GDCtrl(Node.μ,α=α,θ=θ) : SC=GDCtrl(Node.ν,α=α,θ=θ)
            Rwd = SC[2]
            actn = actn, μ, SC[3], SC[4]
        end
    else
        data[1] ? actn = "A1" : actn = "A2"

        μ, Rwd = taskRead(Node.state.name,data[1:2])
        Node.state.R = Rwd
        Node.state.Q = modelledQUpdate(Node,actn,μ,α)
        Node.state.T = transitionUpdate(Node,actn,μ,αₜ)

        if Node.state.name == "ξ"
            μ ? SC=GDCtrl(Node.μ,data=data[3:4],α=α,θ=θ) : SC=GDCtrl(Node.ν,data=data[3:4],α=α,θ=θ)
            Rwd = SC[2]
            actn = actn, μ, SC[3], SC[4]
        end
    end

    return Node, Rwd, actn, π
end
