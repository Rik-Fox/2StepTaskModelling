using Plots
pyplot()

function plotSim(f::Function; data::Union{AbstractArray,Nothing}=Nothing(), ana::Union{Matrix,Bool}=false, N::Int=1000, α::Float64=0.5, θ::Float64=5.0 )
    if typeof(data) != Nothing
        N = length(data[:,1])
    end
    Model = f(n=N,data=data,α=α,θ=θ)
    plt_Q,bar_Q,plt_T,bar_T,plt_h,bar_h,plt_arb = Nothing(),Nothing(),Nothing(),Nothing(),Nothing(),Nothing(),Nothing()
    m = "$f"[22:end]

    if Model[3] != nothing
        if ana == true
            anaQ = zeros(8,N)
            for i=1:N-1
                anaQ[1,i+1] = (1-α)*anaQ[1,i] + α*(0.7*anaQ[3,i]+0.3*anaQ[4,i])
                anaQ[2,i+1] = (1-α)*anaQ[2,i] + α*(0.3*anaQ[3,i]+0.7*anaQ[4,i])
                anaQ[3,i+1] = (1-α)*anaQ[3,i] + α*(findmax([anaQ[5,i] anaQ[6,i]])[1])
                anaQ[4,i+1] = (1-α)*anaQ[4,i] + α*(findmax([anaQ[7,i] anaQ[8,i]])[1])
                anaQ[5,i+1] = (1-α)*anaQ[5,i] + α*0.8
                anaQ[6,i+1] = (1-α)*anaQ[6,i] + α*0.0
                anaQ[7,i+1] = (1-α)*anaQ[7,i] + α*0.2
                anaQ[8,i+1] = (1-α)*anaQ[8,i] + α*0.0
            end
            plot!(anaQ[1,:],label="Analytic ξ.A1",color="blue",linestyle=:dash)
            plot!(anaQ[2,:],label="Analytic ξ.A2",color="orange",linestyle=:dash)
            #plot!(anaQ[3,:],label="Analytic μ.A1",color="green",linestyle=:dash)
            #plot!(anaQ[4,:],label="Analytic ν.A1",color="magenta",linestyle=:dash)
        elseif typeof(ana) == Matrix{Float64}
            anaQ = zeros(8,N)
            anaQ[5:8,:] = ana'
            for i=1:N-1
                anaQ[1,i+1] = (1-α)*anaQ[1,i] + α*(0.7*anaQ[3,i]+0.3*anaQ[4,i])
                anaQ[2,i+1] = (1-α)*anaQ[2,i] + α*(0.3*anaQ[3,i]+0.7*anaQ[4,i])
                anaQ[3,i+1] = (1-α)*anaQ[3,i] + α*(findmax([anaQ[5,i] anaQ[6,i]])[1])
                anaQ[4,i+1] = (1-α)*anaQ[4,i] + α*(findmax([anaQ[7,i] anaQ[8,i]])[1])
            end
            plot!(anaQ[1,:],label="Analytic ξ.A1",color="blue",linestyle=:dash)
            plot!(anaQ[2,:],label="Analytic ξ.A2",color="orange",linestyle=:dash)
            #plot!(anaQ[3,:],label="Analytic μ.A1",color="green",linestyle=:dash)
            #plot!(anaQ[4,:],label="Analytic ν.A1",color="magenta",linestyle=:dash)
        else
            anaQ = zeros(4,N)
        end

        plt_Q = plot(Model[3][1,:],label="ξ.A1",color="blue",ylims = (0, 1))
        plot!(Model[3][2,:],label="ξ.A2",color="orange")
        plot!(Model[3][3,:],label="μ.A1",color="green")
        plot!(Model[3][4,:],label="μ.A2")
        plot!(Model[3][5,:],label="ν.A1",color="magenta")
        plot!(Model[3][6,:],label="ν.A2")

        title!("$m Time Series of Q values")
        xaxis!("Number of iterations")
        yaxis!("Q(s,a)")

        bar_Q = bar([Model[1].state.Q.A1, Model[1].state.Q.A2, Model[1].μ.state.Q.A1, Model[1].μ.state.Q.A2, Model[1].ν.state.Q.A1, Model[1].ν.state.Q.A2],legend=false,ylims = (0, 1));
        title!("$m Final Q values")
        xaxis!("Action(s) Taken")
        xticks!((1:6),["ξ.A1", "ξ.A2", "μ.A1", "μ.A2", "ν.A1", "ν.A2"])
        yaxis!("Q(s,a)")
    end

    if Model[1].state.T != nothing

        plt_T = plot(Model[4][1,:],label="ξ.A1",color="blue",ylims = (0, 1))
        plot!(Model[4][2,:],label="ξ.A2",color="orange")
        plot!(Model[4][3,:],label="μ.A1",color="green")
        plot!(Model[4][4,:],label="μ.A2")
        plot!(Model[4][5,:],label="ν.A1",color="magenta")
        plot!(Model[4][6,:],label="ν.A2")
        title!("$m Time Series of Transition Model")

        xaxis!("Number of iterations")
        yaxis!("T(s,a,s')")

        bar_T = bar([Model[1].state.T.A1, Model[1].state.T.A2, Model[1].μ.state.T.A1, Model[1].μ.state.T.A2, Model[1].ν.state.T.A1, Model[1].ν.state.T.A2],legend=false,ylims = (0, 1));
        title!("$m Final Transition Model")
        xaxis!("Action(s) Taken")
        xticks!((1:6),["ξ.A1", "ξ.A2", "μ.A1", "μ.A2", "ν.A1", "ν.A2"])
        yaxis!("T(s,a,s')")
    end

    if Model[5] != nothing
        plt_h = plot(Model[5][1,:],label="ξ.A1",color="blue",ylims = (0, 1))
        plot!(Model[5][2,:],label="ξ.A2",color="orange")
        plot!(Model[5][3,:],label="μ.A1",color="green")
        plot!(Model[5][4,:],label="μ.A2")
        plot!(Model[5][5,:],label="ν.A1",color="magenta")
        plot!(Model[5][6,:],label="ν.A2")

        title!("$m Time Series of values")
        xaxis!("Number of iterations")
        yaxis!("h(s,a)")

        bar_h = bar([Model[1].state.h.A1, Model[1].state.h.A2, Model[1].μ.state.h.A1, Model[1].μ.state.h.A2, Model[1].ν.state.h.A1, Model[1].ν.state.h.A2],legend=false,ylims = (0, 1));
        title!("$m Final Habit values")
        xaxis!("Action(s) Taken")
        xticks!((1:6),["ξ.A1", "ξ.A2", "μ.A1", "μ.A2", "ν.A1", "ν.A2"])
        yaxis!("h(s,a)")
    end
    if Model[1].state.T != nothing && Model[4] != nothing
        plt_arb = plot(Model[6],ylims = (0, 1))
        title!("Probability over time of Goal Directed Controller being chosen")
        xaxis!("Number of iterations")
        yaxis!("Probability of Goal Directed Controller being chosen")
    end

    return Model, plt_Q, bar_Q, plt_T, bar_T, plt_h, bar_h, plt_arb
end

function plotData(exData::AbstractArray,exRwdProb::AbstractArray; α::Float64=0.5, θ::Float64=5.0)
    habitSimResults = plotSim(runHabit,data=exData,ana=exRwdProb,α=α)
    MFSimResults = plotSim(runMF,data=exData,ana=exRwdProb,α=α)
    MBSimResults = plotSim(runMB,data=exData,ana=exRwdProb,α=α)
    GDSimResults = plotSim(runGD,data=exData,ana=exRwdProb,α=α)
    HWVSimResults = plotSim(runHWV,data=exData,ana=exRwdProb,α=α)

    plth = habitSimResults[2]
    pltf = MFSimResults[2]
    pltb = MBSimResults[2]
    pltg = GDSimResults[2]
    pltw = HWVSimResults[2]

    pltQ = plot(plth,pltf,pltb,pltg,pltw,size=(1000,600),legend=false)

    pltb = MBSimResults[4]
    pltg = GDSimResults[4]
    pltw = HWVSimResults[4]

    pltT = plot(pltb,pltg,pltw,size=(1000,600))

    plth = habitSimResults[6]
    pltw = HWVSimResults[6]

    pltH = plot(plth,pltw,size=(1000,600))

    return pltQ, pltT, pltH
end

function plotAllRaw(; n::Int=1000, α::Float64=0.5, θ::Float64=5.0 )
    ## Raw Models
    habit = plotSim(runHabit,N=n,α=α,θ=θ)
    MF = plotSim(runMF,N=n,α=α,θ=θ)
    MB = plotSim(runMB,N=n,α=α,θ=θ)

    return habit, MF, MB
end

function plotAllBlend(; n::Int=1000, α::Float64=0.5, θ::Float64=5.0 )
    GD = plotSim(runGD,N=n,α=α,θ=θ)
    HWV = plotSim(runHWV,N=n,α=α,θ=θ)

    return GD, HWV
end
