using Plots, Revise, CSV, DataFrames
using Distributions
push!(LOAD_PATH, pwd())
using AgentTreeModels
theme(:wong)
pyplot()

agent = buildAgent(2,Trans=true)
agent_habit = buildAgent(2,Trans=true,habit=true)
#
cleanData = groupby(DataFrame(CSV.File("/home/rfox/Project_MSc/data/Subj43.csv", delim=',')), :Flex0_or_Spec1)[1]
exData = [t==1 for t in [cleanData.First_Choice [t==0.3 for t in cleanData.Transition_Prob] cleanData.Second_Choice cleanData.Reward]]
test=Nothing()
plt = plotSim(runSim_mixed(buildAgent(2,Trans=true),100,0.5,α=0.5,β=15.0))
for i=1:100
    global test = agentCtrller_mixed(agent,testData[i,:],λ=0.5)
end
plt[3]

######################################################

#DAW has cached with trans but eligibility? and tree search, i.e DYNA: no mixing
trial_α = collect(0.0:1/99:1.0)
trial_β = collect(0.0:25/99:25)
##### model free
testData = createData(agent,α=0.5,β=15.0)
Post_MF = MLE(agent,testData)
heatmap(trial_β,trial_α,Post_MF)
#Marginals
ind_MF = findmax(Post_MF)
plot(plot(trial_α,Post_MF[:,ind_MF[2][2]]), plot(trial_β,Post_MF[ind_MF[2][1],:]),layout=(2,1))


##### model based
M=10
testData = createData(agent,M,α=0.5,β=15.0)
function MLEavg(Post_MB,testData,M)
    n = 100
    Post = Array{Array{Float64,2},1}(undef,n)
    @time for i = 1:n
        agent = buildAgent(2,Trans=true)
        Post[i] = MLE(agent,testData,M)
    end

    Post_MB = sum(Post)/length(Post)

    return Post_MB
end

Post_MB = MLEavg(zeros(100,100),testData,M)
Post_MB = MLE(agent,testData,M)
heatmap(trial_β,trial_α,Post_MB)
surface(trial_β,trial_α,Post_MB)
P = Post_MB
#Marginals
ind_MB = findmax(P)
plot(plot(trial_α,P[:,ind_MB[2][2]],yaxis=:log), plot(trial_β,P[ind_MB[2][1],:],yaxis=:log),layout=(2,1))
# plot(trial_α,log.(P[:,ind_MB[2][2]]))
# plot!(trial_α,logpdf.(Beta(0.0005,0.0001),trial_α))
#
#
# plot(trial_β,log.(P[ind_MB[2][1],:]))

#DB has cached no trans and DYNA with updating Trans: mixing
w=0.5
ϵ_cut = 0.1
testData = createData_mixed(agent,w,ϵ_cut,α=0.5,β=10.0)
Post_DB = MLE_mixed(agent, testData, w)
heatmap(trial_β,trial_α,Post_DB[:,:,75])
#Marginals
ind_DB = findmax(Post_DB)
plot(plot(trial_α,Post_DB[:,ind_DB[2][2]]), plot(trial_β,Post_DB[ind_DB[2][1],:]),layout=(2,1))


#Miller has valueless and DYNA with updating Trans: mixing

w=0.33
w1=0.33
w2=0.34
ϵ_cut = 0.1

testData = createData_mixed(agent_habit, w, w1, w2, ϵ_cut, α=0.05, β=10.0)
Post_HWV = MLE_mixed(agent_habit,testData,VF)
heatmap(trial_β,trial_α,Post_HWV)
#Marginals
ind_HWV = findmax(Post_HWV)
plot(plot(trial_α,Post_HWV[:,ind_HWV[2][2]]), plot(trial_β,Post_HWV[ind_HWV[2][1],:]),layout=(2,1))
