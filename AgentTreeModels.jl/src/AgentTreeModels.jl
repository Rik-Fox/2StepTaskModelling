module AgentTreeModels

export Actions, State, DecisionTree, buildAgent
export agentCtrller, runSim, createData
export plotSim, softMax
export log_likelihood

#Hold values for each action
mutable struct Actions{T<:AbstractFloat}
A1      ::T
A2      ::T
end

# Define a simple composite datatype to hold an Q value and the probablistic movement
mutable struct State{T1<:AbstractString, T2<:Actions, T3<:Union{Actions, Nothing}}
name    ::T1
Q       ::T2
T       ::T2
h       ::T3
R       ::T2
e       ::T2
end

# Define the Tree data type for required task
mutable struct DecisionTree{T1<:State}
state   ::T1
μ       ::Union{DecisionTree,Nothing}
ν       ::Union{DecisionTree,Nothing}
end

function buildAgent(steps::Int64; habit::Bool=false, RH::Bool=false)

    Q = Actions(0.0,0.0)
    T = Actions(0.5,0.5)
    habit ? h = Actions(0.0,0.0) : h = Nothing()
    R = Actions(0.0,0.0)
    e = Actions(0.0,0.0)
    if steps == 1
        RH ? name = "ν" : name = "μ"
        Agent = DecisionTree( State(name,Q,T,h,R,e),Nothing(),Nothing())
    else
        Agent = DecisionTree( State("ξ",Q,T,h,R,e), buildAgent(steps-1, habit=habit), buildAgent(steps-1, habit=habit, RH=true) )
    end

    return Agent
end

function rwd(p::Float64)
    if rand() < p
        return 1.0
    else
        return 0.0
    end
end

include(joinpath("functions", "transitionUpdate.jl"))
include(joinpath("functions", "replacetraceUpdate.jl"))
include(joinpath("functions", "rwdUpdate.jl"))
include(joinpath("functions", "habitUpdate.jl"))
include(joinpath("functions", "modelUpdate.jl"))
include(joinpath("functions", "agentUpdate.jl"))
include(joinpath("functions", "askEnviron.jl"))
include(joinpath("functions", "softMax.jl"))
include(joinpath("functions", "agentCtrller.jl"))
include(joinpath("functions", "compareVar.jl"))
include(joinpath("functions", "createData.jl"))
include(joinpath("functions", "runSim.jl"))
include(joinpath("functions", "plotSim.jl"))
include(joinpath("functions", "log_likelihood.jl"))

end
