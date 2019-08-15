using Plots, Revise, CSV, DataFrames
push!(LOAD_PATH, pwd())
using AgentTreeModels
theme(:wong)
pyplot()

function TEST()
    plt = plot()
    thetas = collect{Float64}(1:1:25)
    runtimes = zeros(length(thetas))
    peak = zeros(length(thetas))
    for i=1:length(thetas)
        agent = buildAgent(2,Qvalue=false,habit=true)
        θ = thetas[i]
        testData = createData(agent,α=0.5,θ=θ)

        Esti = @timed MLE(agent,testData)
        Post = Esti[1]
        runtimes[i] = Esti[2]

        ind = findmax(Post)
        peak[i] = ind[2][1]
        plot!(collect(0:25/99:25),Post[ind[2][1],:],label="θ=$θ")
    end
    return plt, runtimes, peak
end
x, y =Nothing(), Nothing()
x, y, peaks = TEST()

plot(peaks)
savefig(x,"thetaRange.png")
print(sum(y)/length(y))
plot(x)

# agent = buildAgent(2)
#
# cleanData = groupby(DataFrame(CSV.File("/home/rfox/Project_MSc/data/Subj43.csv",delim=',')), :Flex0_or_Spec1)[1]
# exData = [t==1 for t in [cleanData.First_Choice [t==0.3 for t in cleanData.Transition_Prob] cleanData.Second_Choice cleanData.Reward]]
# testData = createData(agent,α=0.5,θ=5.0)
#
# # ### param ranges
# trial_α=collect(0.0:1/99:1)
# trial_θ=collect(0:10/99:10)
# #
# # ### prior
# # #using Distributions
# # #prior = ones(length(trial_α))*pdf(Normal(3,1),trial_θ)'
# # prior = ones(length(trial_α),length(trial_θ))
#
# plt = plot()
# plot!(trial_θ,Post[ind[2][1],:])
#
# L, Post = MLE(agent,testData)
# #heatmap(trial_θ,trial_α,Post)
# #surface(trial_θ,trial_α,Post)
#
# #Marginals
# ind = findmax(Post)
# #plot(plot(trial_α,Post[:,ind[2][2]]),plot(trial_θ,Post[ind[2][1],:]),layout=(2,1))
