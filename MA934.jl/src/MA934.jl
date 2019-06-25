module MA934

export KeyValuePair, LinkedList, prepend, append, buildlist, FenwickTree, buildFTree


# Define a simple composite datatype to hold an (Int64, Float64) key-value pair
mutable struct KeyValuePair
    key::Int64
    value::Float64   
end

# Define list type
mutable struct LinkedList
    data::KeyValuePair
    next::Union{LinkedList,Nothing}
end

# Define the FenwickTree data type
mutable struct FenwickTree
    data::KeyValuePair
    left::Union{FenwickTree, Nothing}
    right::Union{FenwickTree, Nothing}
end

# Include the files containing the code for various functions operating on these data types
include(joinpath("functions", "LinkedList.jl"))
include(joinpath("functions", "FenwickTree.jl"))
end
