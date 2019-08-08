using Random, Plots, Revise, CSV, DataFrames
using Distributions
push!(LOAD_PATH, pwd())
using StepTaskModelling
theme(:wong)
pyplot()

cleanData = groupby(DataFrame(CSV.File("/home/rfox/Project_MSc/data/Subj43.csv",delim=',')), :Flex0_or_Spec1)[1]

exData = [t==1 for t in [cleanData.First_Choice [t==0.3 for t in cleanData.Transition_Prob] cleanData.Second_Choice cleanData.Reward]]

testData = taskCreateData(MFCtrl,α=0.6,θ=3.0)

function LogLikeli(L,modelFit,data)
    p = modelFit[2],modelFit[3]
    for i=1:length(p[1])
        if data[i,1]
            L += log(p[1][i])
        else
            L += log(1-p[1][i])
        end
        if data[i,3]
            L += log(p[2][i])
        else
            L += log(1-p[2][i])
        end
    end
    return L
end

function MLE(f::Function, data; trial_α=collect(0.0:1/99:1.0), trial_θ=collect(0:10/99:10), prior=ones(length(trial_α),length(trial_θ)))

    L = zeros(length(trial_α),length(trial_θ))

    for i = 1:length(trial_α)
        for j = 1:length(trial_θ)
            model = f(data=data,α=trial_α[i],θ=trial_θ[j])
            L[i,j] = LogLikeli(L[i,j],model[2],data)
        end
    end
    Post = exp.(L .* prior)
    Post = Post/sum(Post)

    return L, Post
end

model = runMF(data=testData)
plt = plotSim(runMB,data=exData)
plt[3]

trial_α=collect(0.05:1/99:1)
trial_θ=collect(1:10/99:10)
#prior = ones(length(trial_α))*pdf(Normal(3,1),trial_θ)'
prior = ones(length(trial_α),length(trial_θ))
L, Post = MLE(runMF,testData,prior=prior, trial_α=collect(0.05:1/99:1),trial_θ=collect(1:10/99:10))

heatmap(trial_θ,trial_α,Post)
surface(trial_θ,trial_α,Post)

#Marginals
ind = findmax(Post)

plot(plot(trial_α,Post[:,ind[2][2]]),plot(trial_θ,Post[ind[2][1],:]),layout=(2,1))
#vline!([μ_hat])
