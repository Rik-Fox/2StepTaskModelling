using StatsPlots, DataFrames, DelimitedFiles, Pkg, StatsBase
pyplot()
theme(:wong)

Daw = readdlm("Results/res_daw.csv")
daw = DataFrame(α=Daw[:,1], β1=Daw[:,2], β2=Daw[:,3], λ=Daw[:,4], ηₜ=Daw[:,5], ηᵣ=Daw[:,6], MaxL=log.(Daw[:,7]), BIC=log(300)*6 .- 2 .*log.(Daw[:,7]))
Dez = readdlm("Results/res_dez.csv")
dez = DataFrame(α1=Dez[:,1], α2=Dez[:,2], β1=Dez[:,3], β2=Dez[:,4], λ=Dez[:,5], ηₜ=Dez[:,6], κ=Dez[:,7], w=Dez[:,8], MaxL=log.(Dez[:,9]), BIC=log(300)*8 .- 2 .*log.(Dez[:,9]))
Mil = readdlm("Results/res_mil.csv")
mil = DataFrame(α=Mil[:,1], β1=Mil[:,2], β2=Mil[:,3], ηₜ=Mil[:,4], ηᵣ=Mil[:,5], wₒ=Mil[:,6], wₕ=Mil[:,7], wᵧ=Mil[:,8], MaxL=log.(Mil[:,9]), BIC=log(300)*8 .- 2 .*log.(Mil[:,9]))


# dw = -log.(Daw[:,7])
# dz = -log.(Dez[:,9])
# ml = -log.(Mil[:,9])
dw = Daw[:,7]
dz = Dez[:,9]
ml = Mil[:,9]
data = [dw dz ml]
plot(data)
daw_σ = sqrt((sum(dw .- mean(dw)).^2)/112)
dez_σ = sqrt((sum(dz .- mean(dz)).^2)/112)
mil_σ = sqrt((sum(ml .- mean(ml)).^2)/112)
# yerror = [mean([x for x in dw if x < mean(dw)]) mean([x for x in dz if x < mean(dz)]) mean([x for x in ml if x < mean(ml)]); mean([x for x in dw if x > mean(dw)]) mean([x for x in dz if x > mean(dz)]) mean([x for x in ml if x > mean(ml)])]
dez.λ

import PyPlot
yerror = [daw_σ,dez_σ,mil_σ]
plot(title="a")
@df daw plot!(cornerplot([:MaxL :α :β1 :β2 :λ :ηₜ :ηᵣ], compact=true, size = (1000, 800)),tickfont=f,titlefont=f2,guidefont=f1,legendfont=f)

using Pkg
Pkg.add("Blink")

using Interact, Blink
# WebIO.webio_serve(page("/", req -> ui), rand(8000:9000))
ui = button()
display(ui)
w = Window()
body!(w, ui);


title!()
savefig("Plots/CorrDaw.png")
body!(w, @df dez cornerplot([:MaxL :α1 :α2 :β1 :β2 :λ :ηₜ :κ :w], compact=true, size = (1000, 800)))
savefig("Plots/CorrDez.png")
@df mil cornerplot([:MaxL :α :β1 :β2 :ηₜ :ηᵣ :wₒ :wₕ :wᵧ], compact=true, size = (1000, 800))
savefig("Plots/CorrMil.png")
dat = [daw.MaxL dez.MaxL mil.MaxL]
best = findmax(dat, dims=2)
worst=findmin(dat, dims=2)
x = best[2]
daw_win_ind = [ best[2][i][1] for i =1:length(best[2]) if best[2][i][2] == 1]
dez_win_ind = [ best[2][i][1] for i =1:length(best[2]) if best[2][i][2] == 2]
mil_win_ind = [ best[2][i][1] for i =1:length(best[2]) if best[2][i][2] == 3]

daw_win = [ best[1][i][1] for i =1:length(best[2]) if best[2][i][2] == 1]
dez_win = [ best[1][i][1] for i =1:length(best[2]) if best[2][i][2] == 2]
mil_win = [ best[1][i][1] for i =1:length(best[2]) if best[2][i][2] == 3]

daw_worse_ind = [ worst[2][i][1] for i =1:length(best[2]) if worst[2][i][2] == 1]
dez_worse_ind = [ worst[2][i][1] for i =1:length(best[2]) if worst[2][i][2] == 2]
mil_worse_ind = [ worst[2][i][1] for i =1:length(best[2]) if worst[2][i][2] == 3]

daw_worse = [ worst[1][i][1] for i =1:length(best[2]) if worst[2][i][2] == 1]
dez_worse = [ worst[1][i][1] for i =1:length(best[2]) if worst[2][i][2] == 2]
mil_worse = [ worst[1][i][1] for i =1:length(best[2]) if worst[2][i][2] == 3]


Best = vcat(daw_win, dez_win)# mil_win]
Best = vcat(Best,mil_win)

Worst = vcat(daw_worse, dez_worse)# mil_win]
Worst = vcat(Worst,mil_worse)

Diff = Best .- Worst

open("Data/daw_describe.csv","w") do io
    writedlm(io, [des_daw.variable des_daw.mean des_daw.median])
end

open("Data/dez_describe.csv","w") do io
    writedlm(io, [des_dez.variable des_dez.mean des_dez.median])
end

open("Data/mil_describe.csv","w") do io
    writedlm(io, [des_mil.variable des_mil.mean des_mil.median])
end

[des_daw.variable des_daw.mean des_daw.median]

des_daw = describe(daw)
des_dez = describe(dez)
des_mil = describe(mil)

println(des_daw)
length(daw_win) + length(dez_win) + length(mil_win)

daw_win = DataFrame(daw.MaxL[daw_win_ind])
f=font(12,"Helvetica")
f1=font(15,"Helvetica")
f2=font(18,"Helvetica")
density(daw_win, label="Varience Comparision Model", fill=true, alpha=0.35,tickfont=f,titlefont=f2,guidefont=f1,legendfont=f)
density!(dez_win, label="Proportional Model", fill=true, alpha=0.35)
density!(mil_win, label="Valueless Model", fill=true, alpha=0.35)
xlabel!("Log Likelihood")
ylabel!("Normalised Frequency of Occurance")
title!("Distribution of Likelihoods for Winning Models")
savefig("Plots/LikeliDensity.png")

@df daw density(:MaxL)
@df dez density!(:MaxL)
@df mil density!(:MaxL)


bar(daw_win,yaxis=:flip)
bar!(dez_win_ind,dez_win)
bar!(mil_win_ind,mil_win)

histogram(daw_win)
histogram!(dez_win)
histogram!(mil_win_ind,mil_win)


groupedbar(daw_win, dez_win, )

DataBIC = [daw.BIC dez.BIC mil.BIC]
Data = [Daw[:,7] Dez[:,9] Mil[:,9]]
Data_lrt = [DataBIC[:,1]./DataBIC[:,2] DataBIC[:,1]./DataBIC[:,3] DataBIC[:,2]./DataBIC[:,3]]
boxplot(["Varience Compare" "Proportional" "Value-less"], DataBIC, legend=false,tickfont=f,titlefont=f2,guidefont=f1,legendfont=f)
title!("Model Likelihood Ratios")
ylabel!("BIC")
xlabel!("Model Variant")
savefig("Plots/BIC.png")
b_best - b_next


- confusion matrix from sim agents
scatter(["daw","dez","mil"], [mean(dw),mean(dz),mean(ml)], yerr = yerror.*2)
scatter(Mil[:,6],Mil[:,7],Mil[:,8],zcolor=true)
corrplot(Mil[:,6],Mil[:,8])
plot!(cor(Mil[:,6],Mil[:,8]))
scatter(sort([dw dz ml],dims=1), yaxis=:log)
histogram(-log.(data), bins=45)

histogram(-log.(Daw[:,7]), bins=45)

-log(10e-20)

sum()
mean([x for x in Mil[:,9] if x < mean(Mil[:,9])])
