module CustomStructs

export Actions, State, DecisionTree

#Hold Q values for each action
mutable struct Actions
A1      ::Float64
A2      ::Float64
end
# Define a simple composite datatype to hold an Q value and the probablistic movement
mutable struct State
Q       ::Actions
T       ::Union{Actions,Nothing}
h       ::Actions
R       ::Float64
end

# Define the Tree data type for required task
mutable struct DecisionTree
state   ::State
A1      ::Union{DecisionTree, Float64}
A2      ::Union{DecisionTree, Float64}
end

end
