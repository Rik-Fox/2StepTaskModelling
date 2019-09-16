function softMax(A::AbstractArray; β::Float64=5.0)
    a = A[1]
    p = exp(β*a)/sum(exp.(β.*A))
    return p
end

function softMax(A::AbstractArray, κ::Float64, prev::String; β::Float64=5.0)
    if prev == "A1"
        K = [κ, 0.]
        k = κ
    else
        K = [0., κ]
        k = 0.
    end
    a = A[1]
    p = exp((β*a)+k)/sum(exp.( (β*A)+K ))
    return p
end
