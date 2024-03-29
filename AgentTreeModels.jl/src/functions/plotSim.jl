using Plots
pyplot()

function plotSim(Model::T1; ana::Union{Matrix,Bool} = false, α::T = 0.5, θ::T = 5.0, ctrller::String = "MDP") where {T <: Float64,T1 <: Tuple}
    plt_R, bar_R, plt_h, bar_h, plt_arb = Nothing(), Nothing(), Nothing(), Nothing(), Nothing()

    N = length(Model[3][1][1,:])
    if Model[3] != nothing


        plt_Q = plot(Model[3][1][1,:], label = "ξ.A1", color = "blue", ylims = (0, 1))
        plot!(Model[3][1][2,:], label = "ξ.A2", color = "orange")
        plot!(Model[3][1][3,:], label = "μ.A1", color = "green")
        plot!(Model[3][1][4,:], label = "μ.A2")
        plot!(Model[3][1][5,:], label = "ν.A1", color = "magenta")
        plot!(Model[3][1][6,:], label = "ν.A2")

        title!("$ctrller Time Series of Q values")
        xaxis!("Number of iterations")
        yaxis!("Q(s,a)")

        if ana == true
            anaQ = zeros(8, N)
            for i = 1:N - 1
                anaQ[1,i + 1] = (1 - α) * anaQ[1,i] + α * (0.7 * anaQ[3,i] + 0.3 * anaQ[4,i])
                anaQ[2,i + 1] = (1 - α) * anaQ[2,i] + α * (0.3 * anaQ[3,i] + 0.7 * anaQ[4,i])
                anaQ[3,i + 1] = (1 - α) * anaQ[3,i] + α * (findmax([anaQ[5,i] anaQ[6,i]])[1])
                anaQ[4,i + 1] = (1 - α) * anaQ[4,i] + α * (findmax([anaQ[7,i] anaQ[8,i]])[1])
                anaQ[5,i + 1] = (1 - α) * anaQ[5,i] + α * 0.8
                anaQ[6,i + 1] = (1 - α) * anaQ[6,i] + α * 0.0
                anaQ[7,i + 1] = (1 - α) * anaQ[7,i] + α * 0.2
                anaQ[8,i + 1] = (1 - α) * anaQ[8,i] + α * 0.0
            end
            plot!(anaQ[1,:], label = "Analytic ξ.A1", color = "blue", linestyle = :dash)
            plot!(anaQ[2,:], label = "Analytic ξ.A2", color = "orange", linestyle = :dash)
            #plot!(anaQ[3,:],label="Analytic μ.A1",color="green",linestyle=:dash)
            #plot!(anaQ[4,:],label="Analytic ν.A1",color="magenta",linestyle=:dash)
        elseif typeof(ana) == Matrix{Float64}
            anaQ = zeros(8, N)
            anaQ[5:8,:] = ana'
            for i = 1:N - 1
                anaQ[1,i + 1] = (1 - α) * anaQ[1,i] + α * (0.7 * anaQ[3,i] + 0.3 * anaQ[4,i])
                anaQ[2,i + 1] = (1 - α) * anaQ[2,i] + α * (0.3 * anaQ[3,i] + 0.7 * anaQ[4,i])
                anaQ[3,i + 1] = (1 - α) * anaQ[3,i] + α * (findmax([anaQ[5,i] anaQ[6,i]])[1])
                anaQ[4,i + 1] = (1 - α) * anaQ[4,i] + α * (findmax([anaQ[7,i] anaQ[8,i]])[1])
            end
            plot!(anaQ[1,:], label = "Analytic ξ.A1", color = "blue", linestyle = :dash)
            plot!(anaQ[2,:], label = "Analytic ξ.A2", color = "orange", linestyle = :dash)
            #plot!(anaQ[3,:],label="Analytic μ.A1",color="green",linestyle=:dash)
            #plot!(anaQ[4,:],label="Analytic ν.A1",color="magenta",linestyle=:dash)
        else
            anaQ = zeros(4, N)
        end

        bar_Q = bar([Model[1].state.Q.A1, Model[1].state.Q.A2, Model[1].μ.state.Q.A1, Model[1].μ.state.Q.A2, Model[1].ν.state.Q.A1, Model[1].ν.state.Q.A2], legend = false, ylims = (0, 1));
        title!("$ctrller Final Q values")
        xaxis!("Action(s) Taken")
        xticks!((1:6), ["ξ.A1", "ξ.A2", "μ.A1", "μ.A2", "ν.A1", "ν.A2"])
        yaxis!("Q(s,a)")
    end

    if Model[1].state.T != nothing

        plt_T = plot(Model[3][2][1,:], label = "ξ.A1", color = "blue", ylims = (0, 1))
        plot!(Model[3][2][2,:], label = "ξ.A2", color = "orange")
        plot!(Model[3][2][3,:], label = "μ.A1", color = "green")
        plot!(Model[3][2][4,:], label = "μ.A2")
        plot!(Model[3][2][5,:], label = "ν.A1", color = "magenta")
        plot!(Model[3][2][6,:], label = "ν.A2")
        title!("$ctrller Time Series of Transition Model")

        xaxis!("Number of iterations")
        yaxis!("T(s,a,s')")

        bar_T = bar([Model[1].state.T.A1, Model[1].state.T.A2, Model[1].μ.state.T.A1, Model[1].μ.state.T.A2, Model[1].ν.state.T.A1, Model[1].ν.state.T.A2], legend = false, ylims = (0, 1));
        title!("$ctrller Final Transition Model")
        xaxis!("Action(s) Taken")
        xticks!((1:6), ["ξ.A1", "ξ.A2", "μ.A1", "μ.A2", "ν.A1", "ν.A2"])
        yaxis!("T(s,a,s')")
    end

    if Model[2][1] != nothing

        plt_R = plot(Model[3][3][1,:], label = "ξ.A1", color = "blue", ylims = (0, 1))
        plot!(Model[3][3][2,:], label = "ξ.A2", color = "orange")
        plot!(Model[3][3][3,:], label = "μ.A1", color = "green")
        plot!(Model[3][3][4,:], label = "μ.A2")
        plot!(Model[3][3][5,:], label = "ν.A1", color = "magenta")
        plot!(Model[3][3][6,:], label = "ν.A2")
        title!("$ctrller Time Series of Reward Model")

        xaxis!("Number of iterations")
        yaxis!("R(s)")

        bar_R = bar([Model[1].state.R.A1, Model[1].state.R.A2, Model[1].μ.state.R.A1, Model[1].μ.state.R.A2, Model[1].ν.state.R.A1, Model[1].ν.state.R.A2], legend = false, ylims = (0, 1));
        title!("$ctrller Final Reward Model")
        xaxis!("Action(s) Taken")
        xticks!((1:6), ["ξ.A1", "ξ.A2", "μ.A1", "μ.A2", "ν.A1", "ν.A2"])
        yaxis!("R(s)")
    end

    if Model[1].state.e != nothing

        plt_e = plot(Model[3][4][1,:], label = "ξ.A1", color = "blue", ylims = (0, 1))
        plot!(Model[3][4][2,:], label = "ξ.A2", color = "orange")
        plot!(Model[3][4][3,:], label = "μ.A1", color = "green")
        plot!(Model[3][4][4,:], label = "μ.A2")
        plot!(Model[3][4][5,:], label = "ν.A1", color = "magenta")
        plot!(Model[3][4][6,:], label = "ν.A2")
        title!("$ctrller Time Series of Eligibility Model")

        xaxis!("Number of iterations")
        yaxis!("e(s,a)")

        bar_e = bar([Model[1].state.e.A1, Model[1].state.e.A2, Model[1].μ.state.e.A1, Model[1].μ.state.e.A2, Model[1].ν.state.e.A1, Model[1].ν.state.e.A2], legend = false, ylims = (0, 1));
        title!("$ctrller Final Eligibility Model")
        xaxis!("Action(s) Taken")
        xticks!((1:6), ["ξ.A1", "ξ.A2", "μ.A1", "μ.A2", "ν.A1", "ν.A2"])
        yaxis!("e(s,a)")
    end

    if Model[1].state.h != nothing
        plt_h = plot(Model[3][5][1,:], label = "ξ.A1", color = "blue", ylims = (0, 1))
        plot!(Model[3][5][2,:], label = "ξ.A2", color = "orange")
        plot!(Model[3][5][3,:], label = "μ.A1", color = "green")
        plot!(Model[3][5][4,:], label = "μ.A2")
        plot!(Model[3][5][5,:], label = "ν.A1", color = "magenta")
        plot!(Model[3][5][6,:], label = "ν.A2")

        title!("$ctrller Time Series of values")
        xaxis!("Number of iterations")
        yaxis!("h(s,a)")

        bar_h = bar([Model[1].state.h.A1, Model[1].state.h.A2, Model[1].μ.state.h.A1, Model[1].μ.state.h.A2, Model[1].ν.state.h.A1, Model[1].ν.state.h.A2], legend = false, ylims = (0, 1));
        title!("$ctrller Final Habit values")
        xaxis!("Action(s) Taken")
        xticks!((1:6), ["ξ.A1", "ξ.A2", "μ.A1", "μ.A2", "ν.A1", "ν.A2"])
        yaxis!("h(s,a)")
    end
    if Model[4] != nothing
        plt_arb = plot(Model[4], ylims = (0, 1))
        title!("Probability over time of Goal Directed Controller being chosen")
        xaxis!("Number of iterations")
        yaxis!("Probability of Goal Directed Controller being chosen")
    end

    return plt_Q, bar_Q, plt_T, bar_T, plt_R, bar_R, plt_e, bar_e, plt_h, bar_h, plt_arb
end
