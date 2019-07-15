push!(LOAD_PATH, pwd())
using Random, Plots, StepTaskModelling
pyplot()

habitSimResults = plotSim(runHabit,N=5000,α=0.01)
MFSimResults = plotSim(runMF,N=5000,α=0.01)
MBSimResults = plotSim(runMB,N=5000,α=0.01)
GDSimResults = plotSim(runGD,N=5000,α=0.01)
HWVSimResults = plotSim(runHWV,N=5000,α=0.05)

plot(habitSimResults[6],legend=false)
plot(MFSimResults[2],legend=false)
plot(MBSimResults[2],MBSimResults[4],legend=false)
plot(GDSimResults[2],GDSimResults[4],legend=false)
plot(HWVSimResults[2],HWVSimResults[4],HWVSimResults[6],HWVSimResults[8],legend=false)
