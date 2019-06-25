# Define list type
#mutable struct LinkedList
#    data::KeyValuePair
#    next::Union{LinkedList,Nothing}
#end

# Prepend data to list
function prepend(list::Union{LinkedList,Nothing}, data::KeyValuePair) 
    new = LinkedList(data, Nothing())
    new.next = list
    return new
end

# Append data to list
function append(list::Union{LinkedList, Nothing}, data::KeyValuePair) 
    if(list == Nothing())
        # Base case: this is the end of the list so add new node
        return LinkedList(data, Nothing())
    else
        # Recursive case: Append the data to the remainder of the list
        list.next = append(list.next, data)
        return list
    end
end

# Function to build a list from an array of Pair objects
function buildlist(dataArray::Array{KeyValuePair, 1})
    L = LinkedList(dataArray[1], Nothing())
    for i in 2:length(dataArray)
        L=append(L, dataArray[i])
    end
    return L
end
