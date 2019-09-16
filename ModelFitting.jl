using CSV, DataFrames, DelimitedFiles
using Plots, Revise, Distributed
push!(LOAD_PATH, pwd())
using AgentTreeModels, Optim, BlackBoxOptim
theme(:wong)
pyplot()

# cleanData = groupby(DataFrame(CSV.File("/home/rfox/Project_MSc/data/Subj43.csv", delim = ',')), :Flex0_or_Spec1)[1]
# exData = [t == 1 for t in [cleanData.First_Choice [t == 0.3 for t in cleanData.Transition_Prob] cleanData.Second_Choice cleanData.Reward]]
# open("Data/data_mil.csv","w") do io
#     writedlm(io, exData)
# end
open("Results/res_mil.csv", "w") do io
    n =112
    best = Array{Array{Float64,1},1}(undef, n)
    for i=1:n
        cleanData = groupby(DataFrame(CSV.File("/home/rfox/Project_MSc/data/Subj$i.csv", delim = ',')), :Flex0_or_Spec1)[1]
        exData = [t == 1 for t in [cleanData.First_Choice [t == 0.3 for t in cleanData.Transition_Prob] cleanData.Second_Choice cleanData.Reward]]

        open("Data/data_mil.csv","w") do io
            writedlm(io, exData)
        end

        bbopt = bbsetup(log_likelihood; Method=:xnes, SearchRange=[(0.0, 1.0), (0.0, 25.0), (0.0,25.0), (0.0,1.0), (0.0,1.0), (0.0,10.0), (0.0,10.0), (0.0,10.0)], MaxFuncEvals= 10000, Workers = workers())
        res = bboptimize(bbopt)
        best[i] = vcat(best_candidate(res), exp(-1*best_fitness(res)))
        println("Iteration $i of $n")
    end
    writedlm(io, best)
end

bbopt = bbsetup(log_likelihood; Method=:xnes, SearchRange=[(0.0, 1.0), (0.0, 25.0), (0.0,25.0), (0.0,1.0), (0.0,1.0), (0.0,10.0), (0.0,10.0), (0.0,10.0)], MaxFuncEvals= 10000, Workers = workers())
res = bboptimize(bbopt)

######################################################
x = [0.000000000000000002, .000003]
softMax(x,β=25.)
softMax(x,0.5,"A2",β=25.)
y = vcat(x, α)
α1, α2 = 0.2,0.2
k = 5.0
w = 0.15
y = [1., 2.]
z = [3., 4.]
(α*y)+z
#### DAW MF 0.2, 10.0, 10.0, 0.2, 0.2, 0.2
α, β1, β2, λ, ηₜ, ηᵣ = 0.2, 10.0, 10.0, 0.15, 0.2, 0.2
data_daw = createData(150, α, β1, β2, λ, ηₜ, ηᵣ)
open("synthData/dawData.csv", "w") do io
    writedlm(io, data_daw)
end
model_daw = runSim(150, α, β1, β2, λ, ηₜ, ηᵣ)
model_daw_fit = runSim(data_daw, α, β1, β2, λ, ηₜ, ηᵣ)

X = max.(model_daw_fit[2][2],1 .- model_daw_fit[2][2])
(sum(X)+sum(Y))/(length(X) + length(Y))
Y = max.(model_daw_fit[2][3],1 .- model_daw_fit[2][3])
exp(sum(log.(X)) + sum(log.(y)))

plt_daw = plotSim(model_daw, ana = true)
plt_daw_fit = plotSim(model_daw_fit, ana = true)
p = plt_daw_fit #plt_daw_fit
#@profiler runSim(150, α, Β, λ, ηₜ, ηᵣ)
L = exp(-log_likelihood(θ_daw))

plot(p[1], p[2], p[7], p[8], layout = (2, 2), titlefontsize = 12, legend = false)
plot(p[3], p[4], p[5], p[6], layout = (2, 2), titlefontsize = 12, legend = false)

#### Dezfouli & Balliene
α1, α2, β1, β2, λ, ηₜ, k, w = 0.5, 0.5, 7.0, 20.0, 0.0, 0.2, 0.5, 0.55
data_DB = createData(150, α1, α2, β1, β2, λ, ηₜ, k, w)
model_DB = runSim(150, α1, α2, β1, β2, λ, ηₜ, k, w)
model_DB_fit = runSim(exData, α1, α2, β1, β2, λ, ηₜ, k, w)
plt_DB = plotSim(model_DB, ana = true)
plt_DB_fit = plotSim(model_DB_fit, ana = true)

exp(sum(log.(model_DB_fit[2][2])))
θ_dez = [0.5, 0.5, 7.0, 20.0, 0.0, 0.2, 0.5, 0.55]
# θ_dez = convert(Array{BigFloat,1},θ_dez)
L = exp(-log_likelihood(θ_dez))

p = plt_DB #plt_DB_fit
p[1]
plot(p[1], p[2], p[7], p[8], layout = (2, 2), titlefontsize = 12, legend = false)
plot(p[3], p[4], layout = (2, 1), titlefontsize = 12, legend = false)

#### Miller et al
α, β1, β2, ηₜ, ηᵣ, w, w1, w2 = 0.4, 2.0, 2.0, 0.2, 0.2, 2., 1., 10.
θ_mil = [0.4, 2.0, 2.0, 0.2, 0.2, 2., 1., 10.]
L = exp(-log_likelihood(θ_mil))

data_Mill = createData(150, α, β1, β2, ηₜ, ηᵣ, w, w1, w2)
model_Mill = runSim(150, α, β1, β2, ηₜ, ηᵣ, w, w1, w2)
model_Mill_fit = runSim(data_Mill, α, β1, β2, ηₜ, ηᵣ, w, w1, w2)
plt_Mill = plotSim(model_Mill, ana = true)
plt_Mill_fit = plotSim(model_Mill_fit, ana = true)

p = plt_Mill_fit #plt_Mill
plot(p[1], p[2], p[9], p[10], layout = (2, 2), titlefontsize = 12, legend = false)
plot(p[3], p[4], p[5], p[6], layout = (2, 2), titlefontsize = 12, legend = false)
plot(p[11])

########################################################
θ_daw = [0.2, 10.0, 10.0, 0.5, 0.2, 0.2]
θ_daw_TV = [0.5, 7.0, 0.0, 0.2, 0.2, 20]
θ_DB = [0.5, 7.0, 0.0, 0.2, 0.2, 20, 0.55]
θ_Mill = [0.5, 7.0, 0.0, 0.2, 0.2, 20, 0.34, 0.33, 0.33]

log_likelihood(θ_daw)
log(0.9)
log(0.1)
########################################################

using BlackBoxOptim

res = compare_optimizers(log_likelihood; SearchRange = [(0.0, 1.0), (0.0, 25.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)], MaxTime = 3.0)

using Optim

opt = optimize(log_likelihood, θ_daw, 5.0)
##############################################################
