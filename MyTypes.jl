module MyTypes

export Actions, State, DecisionTree, buildAgent, Ctrl

#Hold values for each action
mutable struct Actions{T<:AbstractFloat}
A1      ::T
A2      ::T
end

# Define a simple composite datatype to hold an Q value and the probablistic movement
mutable struct State{T1<:AbstractString, T2<:Union{Actions, Nothing}, T3<:Union{Actions, Nothing}, T4<:Union{Actions, Nothing}, T5<:AbstractFloat}
name    ::T1
Q       ::T2
T       ::T3
h       ::T4
R       ::T5
end

# Define the Tree data type for required task
mutable struct DecisionTree{T1<:State}
state   ::T1
μ       ::Union{DecisionTree,Nothing}
ν       ::Union{DecisionTree,Nothing}
end

function buildAgent(steps::Int64; Qvalue::Bool=true, Trans::Bool=false, habit::Bool=false, RH::Bool=false)

    Qvalue ? Q = Actions(0.0,0.0) : Q = Nothing()
    Trans ? T = Actions(0.5,0.5) : T = Nothing()
    habit ? h = Actions(0.0,0.0) : h = Nothing()
    if steps == 1
        RH ? name = "ν" : name = "μ"
        Agent = DecisionTree( State(name,Q,T,h,0.0),Nothing(),Nothing())
    else
        Agent = DecisionTree( State("ξ",Q,T,h,0.0), buildAgent(steps-1, Qvalue=Qvalue, Trans=Trans, habit=habit), buildAgent(steps-1, Qvalue=Qvalue, Trans=Trans, habit=habit, RH=true) )
    end

    return Agent
end

function Ctrller(Node::DecisionTree{State{String, T2, T2, T1, T}}, data::Nothing; α::T=0.5,θ::T=5.0) where T<:AbstractFloat where T1<:Actions where T2<:Nothing


    π = softMax([Node.state.h.A1, Node.state.h.A2],θ=θ)
    π >= rand() ? actn = "A1" : actn = "A2"

    μ, Rwd = taskEval(Node.state.name,actn)
    Node.state.R = Rwd
    Node.state.h = habitUpdate(Node.state.h,actn,α)
    if Node.state.name == "ξ"
        μ ? SC=habitCtrl(Node.μ,α=α,θ=θ) : SC=MFCtrl(Node.ν,α=α,θ=θ)
        Rwd = SC[2]
        actn = actn, μ, SC[3], SC[4]
    end
    return Node, Rwd, actn, π
end

end
