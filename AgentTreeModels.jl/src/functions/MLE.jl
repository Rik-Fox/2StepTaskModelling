### DAW Models

###model free
function MLE(agent::DecisionTree, data::Array{Bool,2}; incr::T=100.0, β_max::T=25.0) where T<:Float64

    d = incr-1

    trial_α = collect(0.0:1/d:1.0)
    trial_β = collect(0.0:β_max/d:β_max)
    L = zeros(length(trial_α),length(trial_β))

    R = CartesianIndices(L)
    Ifirst, Ilast = first(R), last(R)
    I1 = oneunit(Ifirst)

    @time for I in R

        model = runSim(agent,data,α=trial_α[I[1]],β=trial_β[I[2]])

        L[I] = logLikeli(L[I],model[2],data)

        agent = buildAgent(2,Trans=true)

    end

    Post = exp.(L .* ones(size(L))) # data * prior
    Post = Post/sum(Post)

    return Post
end
## model based
function MLE(agent::DecisionTree, data::Array{Bool,2}, M::Int64; incr::T=100.0, β_max::T=25.0) where T<:Float64

    d = incr-1

    trial_α = collect(0.0:1/d:1.0)
    trial_β = collect(0.0:β_max/d:β_max)
    L = zeros(length(trial_α),length(trial_β))

    R = CartesianIndices(L)

    for I in R
        model = runSim(agent,data,M,α=trial_α[I[1]],β=trial_β[I[2]])
        L[I] = logLikeli(L[I],model[2],data)
        agent = buildAgent(2,Trans=true)
    end

    Post = exp.(L .* ones(size(L)))
    Post = Post/sum(Post)

    return Post   #/sum(exp.(L))
end
### Mixed Models
function MLE_mixed(agent::DecisionTree, data::Array{Bool,2}, wₒ::T, wₕ::T, wᵧ::T; incr::T=100.0, β_max::T=25.0) where T<:Float64

    d = incr-1
    trial_λ = collect(0.0:1/d:1.0)
    trial_α = collect(0.0:1/d:1.0)
    trial_β = collect(0.0:β_max/d:β_max)
    L = zeros(length(trial_α),length(trial_β),length(trial_λ))

    R = CartesianIndices(L)

    for I in R

        model = runSim_mixed(agent, data, wₒ, wₕ, wᵧ, α=trial_α[I[1]], β=trial_β[I[2]], λ=trial_λ[I[3]])
        L[I] += logLikeli(L[I],model[2],data)

        typeof(agent.state.h) == Nothing ? agent = buildAgent(2,Trans=true) : agent = buildAgent(2,Trans=true,habit=true)

    end

    Post = exp.(L .* ones(size(L)))
    Post = Post/sum(Post)

    return Post
end

function MLE_mixed(agent::DecisionTree, data::Array{Bool,2}, w::T; incr::T=100.0, β_max::T=25.0) where T<:Float64

    d = incr-1
    trial_λ = collect(0.0:1/d:1.0)
    trial_α = collect(0.0:1/d:1.0)
    trial_β = collect(0.0:β_max/d:β_max)
    L = zeros(length(trial_α),length(trial_β),length(trial_λ))

    R = CartesianIndices(L)

    @time for I in R

        model = runSim_mixed(agent, data, w, α=trial_α[I[1]], β=trial_β[I[2]], λ=trial_λ[I[3]])
        L[I] += logLikeli(L[I],model[2],data)

        typeof(agent.state.h) == Nothing ? agent = buildAgent(2,Trans=true) : agent = buildAgent(2,Trans=true,habit=true)

    end

    Post = exp.(L .* ones(size(L)))
    Post = Post/sum(Post)

    return Post
end
