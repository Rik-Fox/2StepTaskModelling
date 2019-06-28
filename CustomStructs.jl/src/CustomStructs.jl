module CustomStructs

export State, DecisionTree

    # Define a simple composite datatype to hold an Q value and the probablistic movement
    mutable struct State
        Q       ::Float64
        Prob    ::Float64
        reward  ::Array{Float64,1}
    end

    # Define the Tree data type for required task
    mutable struct DecisionTree
        state   ::State
        A1      ::Union{DecisionTree, Nothing}
        A2      ::Union{DecisionTree, Nothing}
    end

end
