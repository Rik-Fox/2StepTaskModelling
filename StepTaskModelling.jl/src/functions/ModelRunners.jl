using Random, Plots
pyplot()

function runHWV(; HWV = buildStepTask(2,TM=true), n::Int=1000, α::Float64=0.5, θ::Float64=5.0 )
    # pre allocating data arrays
    epochHWV_Q = zeros(6,n)
    epochHWV_T = zeros(6,n)
    epochHWV_h = zeros(6,n)

    #HWV.state.T.A1, HWV.state.T.A2, HWV.A1.state.T.A1, HWV.A1.state.T.A2, HWV.A2.state.T.A1, HWV.A2.state.T.A2 = 0.7, 0.7, 1.0, 1.0, 1.0, 1.0
    ## Param init conditions
    P𝑮 = ones(n+1)
    r𝑮 = 0.0
    h_avg = 0.0

    for i=1:n
        old_R = sum([HWV.state.R HWV.A1.state.R HWV.A2.state.R])
        if rand() < P𝑮[i]
            HWV = GDCtrl(HWV,α=α,θ=θ)
            r𝑮 = sum([HWV.state.R HWV.A1.state.R HWV.A2.state.R]) - old_R
        else
            HWV = habitCtrl(HWV,α=α,θ=θ)
            h_avg = sum([HWV.state.h.A1, HWV.state.h.A2, HWV.A1.state.h.A1, HWV.A1.state.h.A2, HWV.A2.state.h.A1, HWV.A2.state.h.A2])/6

        end
        epochHWV_Q[:,i] = [HWV.state.Q.A1, HWV.state.Q.A2, HWV.A1.state.Q.A1, HWV.A1.state.Q.A2, HWV.A2.state.Q.A1, HWV.A2.state.Q.A2]
        epochHWV_T[:,i] = [HWV.state.T.A1, HWV.state.T.A2, HWV.A1.state.T.A1, HWV.A1.state.T.A2, HWV.A2.state.T.A1, HWV.A2.state.T.A2]
        epochHWV_h[:,i] = [HWV.state.h.A1, HWV.state.h.A2, HWV.A1.state.h.A1, HWV.A1.state.h.A2, HWV.A2.state.h.A1, HWV.A2.state.h.A2]

        rwd_diff = (r𝑮 - sum([HWV.state.R HWV.A1.state.R HWV.A2.state.R])) #GD.state.R - (h.state.R + Gd.state.R)
        P𝑮[i+1] = 1/( 1 + exp(abs(rwd_diff) - abs(h_avg^2)) )
    end

    return HWV, epochHWV_Q, epochHWV_T, epochHWV_h, P𝑮

end

function runHabit(; habit::DecisionTree=buildStepTask(2), n::Int=1000, α::Float64=0.5, θ::Float64=5.0 )
    epochHabit = zeros(6,n)
    epochHabit_Q = zeros(6,n)
    for i=1:n
        habit = habitCtrl(habit,α=α,θ=θ)
        epochHabit[:,i] = [habit.state.h.A1, habit.state.h.A2, habit.A1.state.h.A1, habit.A1.state.h.A2, habit.A2.state.h.A1, habit.A2.state.h.A2]
        epochHabit_Q[:,i] = [habit.state.Q.A1, habit.state.Q.A2, habit.A1.state.Q.A1, habit.A1.state.Q.A2, habit.A2.state.Q.A1, habit.A2.state.Q.A2]
    end

    return habit,epochHabit_Q,Nothing(), epochHabit
end

function runMF(; MF::DecisionTree=buildStepTask(2), n::Int=1000, α::Float64=0.5, θ::Float64=5.0)
    epochMF = zeros(6,n)
    for i=1:n
        MF = MFCtrl(MF,α=α,θ=θ)
        epochMF[:,i] = [MF.state.Q.A1, MF.state.Q.A2, MF.A1.state.Q.A1, MF.A1.state.Q.A2, MF.A2.state.Q.A1, MF.A2.state.Q.A2]
    end

    return MF, epochMF, Nothing(), Nothing()
end

function runMB(; MB::DecisionTree=buildStepTask(2,TM=true), n::Int=1000, α::Float64=0.5, θ::Float64=5.0, TM::AbstractArray = [0.7 0.7 1.0 1.0 1.0 1.0] )
    MB.state.T.A1, MB.state.T.A2, MB.A1.state.T.A1, MB.A1.state.T.A2, MB.A2.state.T.A1, MB.A2.state.T.A2 = TM[1], TM[2], TM[3], TM[4], TM[5], TM[6]
    epochMB_Q = zeros(6,n)
    epochMB_T = zeros(6,n)
    for i=1:n
        MB = MBCtrl(MB,α=α,θ=θ)
        epochMB_Q[:,i] = [MB.state.Q.A1, MB.state.Q.A2, MB.A1.state.Q.A1, MB.A1.state.Q.A2, MB.A2.state.Q.A1, MB.A2.state.Q.A2]
        epochMB_T[:,i] = [MB.state.T.A1, MB.state.T.A2, MB.A1.state.T.A1, MB.A1.state.T.A2, MB.A2.state.T.A1, MB.A2.state.T.A2]
    end

    return MB, epochMB_Q, epochMB_T, Nothing()
end

function runGD(; GD::DecisionTree=buildStepTask(2,TM=true), n::Int=1000, α::Float64=0.5, θ::Float64=5.0 )
    epochGD_Q = zeros(6,n)
    epochGD_T = zeros(6,n)
    for i=1:n
        GD = GDCtrl(GD,α=α,αₜ=0.05,θ=θ)
        epochGD_Q[:,i] = [GD.state.Q.A1, GD.state.Q.A2, GD.A1.state.Q.A1, GD.A1.state.Q.A2, GD.A2.state.Q.A1, GD.A2.state.Q.A2]
        epochGD_T[:,i] = [GD.state.T.A1, GD.state.T.A2, GD.A1.state.T.A1, GD.A1.state.T.A2, GD.A2.state.T.A1, GD.A2.state.T.A2]
    end

    return GD, epochGD_Q, epochGD_T,Nothing()
end

function runAll(; n::Int=1000, α::Float64=0.5, θ::Float64=5.0 )
    ## Raw Models
    habit = runHabit(N=n,α=α,θ=θ)
    MF = runMF(N=n,α=α,θ=θ)
    MB = runMB(N=n,α=α,θ=θ)
    # Blend/variation Models
    GD = runGD(N=n,α=α,θ=θ)
    HWV = runHWV(N=n,α=α,θ=θ)

    return habit, MF, MB, GD, HWV
end

function plotAll(; n::Int=1000, α::Float64=0.5, θ::Float64=5.0 )
    ## Raw Models
    habit = plotSim(runHabit,N=n,α=α,θ=θ)
    MF = plotSim(runMF,N=n,α=α,θ=θ)
    MB = plotSim(runMB,N=n,α=α,θ=θ)
    # Blend/variation Models
    GD = plotSim(runGD,N=n,α=α,θ=θ)
    HWV = plotSim(runHWV,N=n,α=α,θ=θ)

    return habit, MF, MB, GD, HWV

end

function plotSim(f::Function; N::Int=1000, α::Float64=0.5, θ::Float64=5.0 )

    Model = f(n=N,α=α,θ=θ)
    plt_Q,bar_Q,plt_T,bar_T,plt_h,bar_h,plt_arb = Nothing(),Nothing(),Nothing(),Nothing(),Nothing(),Nothing(),Nothing()
    m = "$f"[22:end]

    if Model[2] != nothing

        anaQ = zeros(8,N)
        for i=1:N-1
            anaQ[1,i+1] = (1-α)*anaQ[1,i] + α*(0.7*anaQ[3,i]+0.3*anaQ[4,i])
            anaQ[2,i+1] = (1-α)*anaQ[2,i] + α*(0.3*anaQ[3,i]+0.7*anaQ[4,i])
            anaQ[3,i+1] = (1-α)*anaQ[3,i] + α*(findmax([anaQ[5,i] anaQ[6,i]])[1])
            anaQ[4,i+1] = (1-α)*anaQ[4,i] + α*(findmax([anaQ[7,i] anaQ[8,i]])[1])
            anaQ[5,i+1] = (1-α)*anaQ[5,i] + α*Model[1].A1.A1
            anaQ[6,i+1] = (1-α)*anaQ[6,i] + α*Model[1].A1.A2
            anaQ[7,i+1] = (1-α)*anaQ[7,i] + α*Model[1].A2.A1
            anaQ[8,i+1] = (1-α)*anaQ[8,i] + α*Model[1].A2.A2
        end

        plt_Q = plot(Model[2][1,:],label="A1",ylims = (0, 1))
        plot!(Model[2][2,:],label="A2")
        plot!(Model[2][3,:],label="A1.A1")
        plot!(Model[2][4,:],label="A1.A2")
        plot!(Model[2][5,:],label="A2.A1")
        plot!(Model[2][6,:],label="A2.A2")
        plot!(anaQ[1,:],label="Analytic A1",color="blue",linestyle=:dash)
        plot!(anaQ[2,:],label="Analytic A2",color="orange",linestyle=:dash)
        plot!(anaQ[3,:],label="Analytic A1.A1",color="green",linestyle=:dash)
        plot!(anaQ[4,:],label="Analytic A2.A1",color="magenta",linestyle=:dash)

        title!("$m Time Series of Q values")
        xaxis!("Number of iterations")
        yaxis!("Q(s,a)")

        bar_Q = bar([Model[1].state.Q.A1, Model[1].state.Q.A2, Model[1].A1.state.Q.A1, Model[1].A1.state.Q.A2, Model[1].A2.state.Q.A1, Model[1].A2.state.Q.A2],legend=false,ylims = (0, 1));
        title!("$m Final Q values")
        xaxis!("Action(s) Taken")
        xticks!((1:6),["A1", "A2", "A1.A1", "A1.A2", "A2.A1", "A2.A2"])
        yaxis!("Q(s,a)")
    end

    if Model[1].state.T != nothing

        plt_T = plot(Model[3][1,:],label="A1",ylims = (0, 1))
        plot!(Model[3][2,:],label="A2")
        plot!(Model[3][3,:],label="A1.A1")
        plot!(Model[3][4,:],label="A1.A2")
        plot!(Model[3][5,:],label="A2.A1")
        plot!(Model[3][6,:],label="A2.A2")
        title!("$m Time Series of Transition Model")

        xaxis!("Number of iterations")
        yaxis!("T(s,a,s')")

        bar_T = bar([Model[1].state.T.A1, Model[1].state.T.A2, Model[1].A1.state.T.A1, Model[1].A1.state.T.A2, Model[1].A2.state.T.A1, Model[1].A2.state.T.A2],legend=false,ylims = (0, 1));
        title!("$m Final Transition Model")
        xaxis!("Action(s) Taken")
        xticks!((1:6),["A1", "A2", "A1.A1", "A1.A2", "A2.A1", "A2.A2"])
        yaxis!("T(s,a,s')")
    end

    if Model[4] != nothing
        plt_h = plot(Model[4][1,:],label="A1",ylims = (0, 1))
        plot!(Model[4][2,:],label="A2")
        plot!(Model[4][3,:],label="A1.A1")
        plot!(Model[4][4,:],label="A1.A2")
        plot!(Model[4][5,:],label="A2.A1")
        plot!(Model[4][6,:],label="A2.A2")

        title!("$m Time Series of values")
        xaxis!("Number of iterations")
        yaxis!("h(s,a)")

        bar_h = bar([Model[1].state.h.A1, Model[1].state.h.A2, Model[1].A1.state.h.A1, Model[1].A1.state.h.A2, Model[1].A2.state.h.A1, Model[1].A2.state.h.A2],legend=false,ylims = (0, 1));
        title!("$m Final Habit values")
        xaxis!("Action(s) Taken")
        xticks!((1:6),["A1", "A2", "A1.A1", "A1.A2", "A2.A1", "A2.A2"])
        yaxis!("h(s,a)")
    end
    if Model[1].state.T != nothing && Model[4] != nothing
        plt_arb = plot(Model[5],ylims = (0, 1))
        title!("Probability over time of Goal Directed Controller being chosen")
        xaxis!("Number of iterations")
        yaxis!("Probability of Goal Directed Controller being chosen")
    end

    return Model,plt_Q,bar_Q,plt_T,bar_T,plt_h,bar_h,plt_arb
end
