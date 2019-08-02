push!(LOAD_PATH, pwd())
using Random, Plots, StepTaskModelling, CSV, DataFrames, Pkg
pyplot()

# DataFrame(skipmissing(CSV.File("Events2.csv",delim=','; missingstrings=["", "NA"])))
rawData = DataFrame(CSV.File("/home/rfox/Project_MSc/data/Subj3.csv",delim=','))

cleanData = groupby(rawData,:Flex0_or_Spec1)[1]
names(cleanData)

switch = [t==0.3 for t in cleanData.Transition_Prob]

exActn = Array{Bool,2}([switch cleanData.First_Choice cleanData.Second_Choice])

exRwd = Array{Bool,1}(cleanData.Reward)

exactEx = buildStepTask(2,TM=true)

exActn[4,1]

for i=1:length(switch)

    if (exActn[i,2] == true && exActn[i,1] == false) || (exActn[i,2] == false && exActn[i,1] == true)




experimtentRwdProb = [cleanData.p1 cleanData.p2 cleanData.p3 cleanData.p4]

experimentalActn[34,:]


HWV = plotSim(runHWV,N=10000,α=0.01)

HWV[6]
