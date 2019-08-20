### Transition Models
function MLE(agent::DecisionTree, data::Array{Bool,2}, MB::Bool; trial_α::T3=collect(0:1/99:1), trial_θ::T3=collect(0:25/99:25), prior::Array{Float64,2}=ones(length(trial_α),length(trial_θ))) where {T<:AbstractFloat, T1<:Actions, T2<:Nothing, T3<:AbstractArray}

    L = zeros(length(trial_α),length(trial_θ))

    for i = 1:length(trial_α)
        for j = 1:length(trial_θ)
            model = runSim(agent,data,α=trial_α[i],θ=trial_θ[j],MB=MB)
            L[i,j] = logLikeli(L[i,j],model[2],data)
            agent.state.Q == Nothing ? () : (agent.state.Q = Actions(0.0,0.0); agent.μ.state.Q = Actions(0.0,0.0); agent.ν.state.Q = Actions(0.0,0.0))
            agent.state.T == Nothing ? () : (agent.state.T = Actions(0.0,0.0); agent.μ.state.T = Actions(0.0,0.0); agent.ν.state.T = Actions(0.0,0.0))
        end
    end
    Post = exp.(L .* prior)
    Post = Post/sum(Post)

    return Post
end

### Mixed Models
function MLE(agent::DecisionTree, data::Array{Bool,2},habit::Bool; trial_α::T3=collect(0:1/99:1), trial_θ::T3=collect(0:25/99:25), prior::Array{Float64,2}=ones(length(trial_α),length(trial_θ))) where {T<:AbstractFloat, T1<:Actions, T2<:Nothing, T3<:AbstractArray}

    L = zeros(length(trial_α),length(trial_θ))

    for i = 1:length(trial_α)
        for j = 1:length(trial_θ)
            model = runSim_mixed(agent,data,α=trial_α[i],θ=trial_θ[j])
            L[i,j] = logLikeli(L[i,j],model[2],data)

            habit ? agent = buildAgent(2,Trans=true,habit=true) : buildAgent(2,Trans=true)

        end
    end
    Post = exp.(L .* prior)
    Post = Post/sum(Post)

    return Post
end

### value-less models
# function MLE(agent::DecisionTree{State{String,T2,T2,T1,T1}}, data::Array{Bool,2}; trial_α::T3=collect(0:1/99:1), trial_θ::T3=collect(0:25/99:25), prior::Array{Float64,2}=ones(length(trial_α),length(trial_θ))) where {T<:AbstractFloat, T1<:Actions, T2<:Nothing, T3<:AbstractArray}
#
#     L = zeros(length(trial_α),length(trial_θ))
#
#     for i = 1:length(trial_α)
#         for j = 1:length(trial_θ)
#             model = runSim(agent,data,α=trial_α[i],θ=trial_θ[j])
#             L[i,j] = logLikeli(L[i,j],model[2],data)
#             agent.state.h == Nothing ? () : (agent.state.h =Actions(0.0,0.0); agent.μ.state.h =Actions(0.0,0.0); agent.ν.state.h =Actions(0.0,0.0))
#         end
#     end
#     Post = exp.(L .* prior)
#     Post = Post/sum(Post)
#
#     return Post
# end

### no trans prob Models
# function MLE(agent::DecisionTree{State{String,T1,T2,T2,T1}}, data::Array{Bool,2}; trial_α::T3=collect(0:1/99:1), trial_θ::T3=collect(0:25/99:25), prior::Array{Float64,2}=ones(length(trial_α),length(trial_θ))) where {T<:AbstractFloat, T1<:Actions, T2<:Nothing, T3<:AbstractArray}
#
#     L = zeros(length(trial_α),length(trial_θ))
#
#     for i = 1:length(trial_α)
#         for j = 1:length(trial_θ)
#             model = runSim(agent,data,α=trial_α[i],θ=trial_θ[j])
#             L[i,j] = logLikeli(L[i,j],model[2],data)
#             # agent.state.Q == Nothing ? () : (agent.state.Q =Actions(0.0,0.0); agent.μ.state.Q =Actions(0.0,0.0); agent.ν.state.Q =Actions(0.0,0.0))
#         end
#     end
#     Post = exp.(L .* prior)
#     Post = Post/sum(Post)
#
#     return Post
# end
