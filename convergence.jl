using Plots
theme(:wong)
pyplot()

function converge(alphas::Array{Float64,1}, M::Int64)
    plt = plot()
    for i = 1:length(alphas)
        plot!(log.(cumprod(ones(M).*(1-alphas[i]))))
    end
    return plt
end


M = 5000
alphas = collect(0.003:0.001:0.999)
N128 = log(BigFloat(0.1)^6143) ./ log.(1 .- alphas)
N64 =  log(BigFloat(0.1)^383) ./ log.(1 .- alphas)
N32 = log(BigFloat(0.1)^95) ./ log.(1 .- alphas)
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

X = converge(alphas,M)

x = BigFloat(0.1)
x = x^95
e1 = log(BigFloat(0.1)^95)
e1 = log(BigFloat(0.1)^383)
e1 = log(BigFloat(0.1)^6143)
