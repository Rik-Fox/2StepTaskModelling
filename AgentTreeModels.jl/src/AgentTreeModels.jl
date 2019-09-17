module AgentTreeModels
## not all included functions are exported, but are included (at the end of the script) for as they are dependancies for the exported functions
export Actions, State, DecisionTree, buildAgent
export agentCtrller, runSim, createData
export plotSim, softMax

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
h       ::T3        ##hbait counter
R       ::T2        ## reward model
e       ::T2        ## eligbility for replace trace
end

# Define the Tree data type for required task
mutable struct DecisionTree{T1<:State}
state   ::T1
μ       ::Union{DecisionTree,Nothing}
ν       ::Union{DecisionTree,Nothing}
end

### create an agent that matches structure of two step task
function buildAgent(steps::Int64; habit::Bool=false, RH::Bool=false)

    Q = Actions(0.0,0.0)
    T = Actions(0.5,0.5)
    ## this is nessary for some method differentiations
    habit ? h = Actions(0.0,0.0) : h = Nothing()
    R = Actions(0.0,0.0)
    e = Actions(0.0,0.0)
    if steps == 1
        ## RH simply defines if this state is the common transition or A1 if true and vice versa
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

## this is how i know to include functions in a module, but havn't seen this kind of thing in other package source code, there may be a better way of building these functions or exporting
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

end
