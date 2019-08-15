module AgentTreeModels
export Actions, State, DecisionTree, buildAgent
export agentCtrller, runSim, createData, agentUpdate
export MLE

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

################## Misc Functions ##################################################################
function softMax(A::AbstractArray; θ::Float64=5.0)
    a = A[1]
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
include(joinpath("functions", "agentUpdate.jl"))
include(joinpath("functions", "agentCtrller.jl"))
include(joinpath("functions", "transitionUpdate.jl"))
include(joinpath("functions", "askEnviron.jl"))
include(joinpath("functions", "createData.jl"))
include(joinpath("functions", "runSim.jl"))
include(joinpath("functions", "runSim_mixed.jl"))
include(joinpath("functions", "logLikeli.jl"))
include(joinpath("functions", "MLE.jl"))
end
