#############################################################################################
### didn't end up using but may do when looking for better optimisers and is generally useful

struct MatlabSimplexer{T} <: Optim.Simplexer
    a::T
    b::T
end
MatlabSimplexer(;a = 0.00025, b = 0.05) = MatlabSimplexer(a, b)

function Optim.simplexer(A::MatlabSimplexer, initial_x::AbstractArray{T, N}) where {T, N}
    n = length(initial_x)
    initial_simplex = Array{T, N}[initial_x for i = 1:n+1]
    for j = 1:n
        initial_simplex[j+1][j] += initial_simplex[j+1][j] == zero(T) ? S.b * initial_simplex[j+1][j] : S.a
    end
    initial_simplex
end
