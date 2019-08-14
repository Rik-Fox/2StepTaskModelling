module MyTypes

export Actions, State, DecisionTree, buildAgent

abstract type Agent end

#Hold values for each action
struct Actions{T<:AbstractFloat} <: Agent{AbstractFloat}
A1      ::T
A2      ::T
end

# Define a simple composite datatype to hold an Q value and the probablistic movement
struct State{T1<:AbstractString,T<:Actions,T3<:Nothing}
name    ::T1
Q       ::Actions
T       ::Union{Float64,Nothing}
h       ::Union{Float64,Nothing}
R       ::Float64
end

# Define the Tree data type for required task
struct DecisionTree
state   ::State
μ      ::Union{DecisionTree, Nothing}
ν      ::Union{DecisionTree, Nothing}
end

function buildAgent(steps::Int64;TM::Bool=false,hm::Bool=false,RH::Bool=false)

    Q = Actions(0.0,0.0)
    if steps == 1
        RH ? name = "ν" : name = "μ"
        TM ? T = Actions(0.5,0.5) : T = Nothing()
        hm ? h = Actions(0.0,0.0) : h = Nothing()
        Agent = DecisionTree( State(name,Q,T,h,0.0),Nothing(),Nothing())
    else
        TM ? T = Actions(0.5,0.5) : T = Nothing()
        hm ? h = Actions(0.0,0.0) : h = Nothing()
        Agent = DecisionTree( State("ξ",Q,T,h,0.0), buildAgent(steps-1,TM=TM,hm=hm), buildAgent(steps-1,TM=TM,hm=hm,RH=true) )
    end

    return Agent
end

end
