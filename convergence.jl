###########

using Plots
theme(:wong)
pyplot()

M = 5000
#slight truncation as asymtote to ∞ at 0 and to 0 at 1
alphas = collect(0.003:0.001:0.999)
### using smallest number possible for each precision, but all as bigFloats to aviod as much round off error as possible
N128 = log(BigFloat(0.1)^6143) ./ log.(1 .- alphas)
N64 =  log(BigFloat(0.1)^383) ./ log.(1 .- alphas)
N32 = log(BigFloat(0.1)^95) ./ log.(1 .- alphas)

### f,f1,f2 are pyplot font objects, better way than the ylabel!() etc to set plot fonts
f=font(13,"Helvetica")
f1=font(15,"Helvetica")
f2=font(18,"Helvetica")
plot(alphas,N32,label="Float32",yaxis = :log,tickfont=f,titlefont=f2,guidefont=f1,legendfont=f, linewidth=3.0)
plot!(alphas,N64,label="Float64", linestyle = :dash, linewidth=3.0)
plot!(alphas,N128,label="Float128", linestyle = :dot, linewidth=3.0)
xticks!(collect(0.0:0.1:1.0))
ylabel!("Iterations before reaching O(ϵₘ)")
xlabel!("α")
title!("Iterations until Pseudo Convergence")
savefig("converge.png")
