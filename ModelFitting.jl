using Plots, Revise, CSV, DataFrames
push!(LOAD_PATH, pwd())
using AgentTreeModels
theme(:wong)
pyplot()

agent = buildAgent(2,Trans=true)
agent_habit = buildAgent(2,Trans=true,habit=true)
#
cleanData = groupby(DataFrame(CSV.File("/home/rfox/Project_MSc/data/Subj43.csv", delim=',')), :Flex0_or_Spec1)[1]
exData = [t==1 for t in [cleanData.First_Choice [t==0.3 for t in cleanData.Transition_Prob] cleanData.Second_Choice cleanData.Reward]]

# ### param ranges
trial_α=collect(0.0:1/99:1)
trial_θ=collect(0:25/99:25)

### prior
#using Distributions
#prior = ones(length(trial_α))*pdf(Normal(3,1),trial_θ)'
prior = ones(length(trial_α),length(trial_θ))


######################################################

#DAW has cached with trans but eligibility? and tree search, i.e DYNA: no mixing

##### model free
MB = false
testData = createData(agent,MB,α=0.5,θ=10.0)
Post_MF = MLE(agent,testData,MB)
heatmap(trial_θ,trial_α,Post_MF)
#Marginals
ind_MF = findmax(Post_MF)
plot(plot(trial_α,Post_MF[:,ind_MF[2][2]]), plot(trial_θ,Post_MF[ind_MF[2][1],:]),layout=(2,1))


##### model based
MB=true
testData = createData(agent,MB,α=0.75,θ=10.0)
Post_MB = MLE(agent,testData,MB)
heatmap(trial_θ,trial_α,Post_MB)
#Marginals
ind_MB = findmax(Post_MB)
plot(plot(trial_α,Post_MB[:,ind_MB[2][2]]), plot(trial_θ,Post_MB[ind_MB[2][1],:]),layout=(2,1))

#DB has cached no trans and DYNA with updating Trans: mixing

testData = createData_mixed(agent,α=0.5,θ=10.0)
VF = false
Post_DB = MLE(agent,testData,VF)
heatmap(trial_θ,trial_α,Post_DB)
#Marginals
ind_DB = findmax(Post_DB)
plot(plot(trial_α,Post_DB[:,ind_DB[2][2]]), plot(trial_θ,Post_DB[ind_DB[2][1],:]),layout=(2,1))


#Miller has valueless and DYNA with updating Trans: mixing

testData = createData_mixed(agent_habit,α=0.05,θ=10.0)
VF = true
Post_HWV = MLE(agent_habit,testData,VF)
heatmap(trial_θ,trial_α,Post_HWV)
#Marginals
ind_HWV = findmax(Post_HWV)
plot(plot(trial_α,Post_HWV[:,ind_HWV[2][2]]), plot(trial_θ,Post_HWV[ind_HWV[2][1],:]),layout=(2,1))
