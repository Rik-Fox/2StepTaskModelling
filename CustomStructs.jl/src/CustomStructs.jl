module CustomStructs

export Actions, State, DecisionTree, buildAgent

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

end
