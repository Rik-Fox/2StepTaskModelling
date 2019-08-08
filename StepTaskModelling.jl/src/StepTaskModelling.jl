module StepTaskModelling

export habitCtrl, MFCtrl, MBCtrl, GDCtrl, HWVCtrl
export runHabit, runMF, runMB, runGD, runHWV, runAll
export plotSim, plotAll
export Actions, State, DecisionTree, buildAgent
export softMax, taskCreateData

#Hold values for each action
mutable struct Actions
A1      ::Float64
A2      ::Float64
end
# Define a simple composite datatype to hold an Q value and the probablistic movement
mutable struct State
name    ::String
Q       ::Actions
T       ::Union{Actions,Nothing}
h       ::Union{Actions,Nothing}
R       ::Float64
end

# Define the Tree data type for required task
mutable struct DecisionTree
state   ::State
μ      ::Union{DecisionTree, Nothing}
ν      ::Union{DecisionTree, Nothing}
end

############ Create Agent ###############################################

function buildAgent(steps::Int;TM::Bool=false,hm::Bool=false,R::Bool=false)

    Q = Actions(0.0,0.0)
    if steps == 1
        R ? name = "ν" : name = "μ"
        TM ? T = Actions(0.5,0.5) : T = Nothing()
        hm ? h = Actions(0.0,0.0) : h = Nothing()
        Agent = DecisionTree( State(name,Q,T,h,0.0),Nothing(),Nothing())
    else
        TM ? T = Actions(0.5,0.5) : T = Nothing()
        hm ? h = Actions(0.0,0.0) : h = Nothing()
        Agent = DecisionTree( State("ξ",Q,T,h,0.0), buildAgent(steps-1,TM=TM,hm=hm), buildAgent(steps-1,TM=TM,hm=hm,R=true) )
    end

    return Agent
end

################## Misc Functions ##################################################################
function softMax(A; θ::Float64=5.0)
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

include(joinpath("functions", "ModelControllers.jl"))
include(joinpath("functions", "ModelRunners.jl"))
include(joinpath("functions", "ModelUpdates.jl"))
include(joinpath("functions", "ModelPlotting.jl"))
include(joinpath("functions", "ModelEnvironments.jl"))
end
