# Function to build a Fenwick tree from an array of KeyValuePair objects containing list of real numbers
function buildFTree(dataArray::Array{KeyValuePair, 1})
    if length(dataArray) == 1
        # Base case: if dataArray has lenght 1, we simply return the tree containing this key-value pair
        return FenwickTree(dataArray[1], Nothing(), Nothing())
    else
        # Recursive case: this is the tricky bit. We first compute the sum, S, of the values contained in the elements of
        # dataArray. We then split dataArray into two approximately equal parts accounting for the possibility of an odd
        # number of elements. We then return a tree containin the KeyValuePair (-1, S) in its data field and the two 
        # parts of dataArray in its left and right sub-trees.
        S=0.0
        for i in 1:length(dataArray)
            S+=dataArray[i].value
        end
        k = KeyValuePair(-1, S)
        
        m=floor(Int64, length(dataArray)/2)
        L = dataArray[1:m]
        R = dataArray[m+1:end]
        
        return FenwickTree(k, buildFTree(L), buildFTree(R))
    end
end

        