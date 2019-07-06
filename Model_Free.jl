using Random, Plots

############### SET UP WORLD #########################################
push!(LOAD_PATH, pwd())
using CustomStructs

function buildStepTask(steps;MB=false)

    if steps == 1

        Q = Actions(0.0,0.0)
        if MB == true
            T = Actions(0.0,0.0)
        else
            T = Nothing()
        end
        R = 0.5

        Task = DecisionTree(State(Q,T,R),Nothing(),Nothing())

    else

        Q = Actions(0.0,0.0)
        if MB == true
            T = Actions(0.0,0.0)
        else
            T = Nothing()
        end
        R = 0

        Task = DecisionTree(State(Q,T,R), buildStepTask(steps-1,MB=MB), buildStepTask(steps-1,MB=MB))
    end

    return Task

end

findall(x->x==findmax(A)[1], A)

habit = buildStepTask(3)
MF_model = buildStepTask(3)
MB_model = buildStepTask(3, MB=true)

########################################################################

function softMax(a, A; θ::Float64=1.0)
    p = exp(θ*a)/sum(exp(θ*A))
    return p
end

function Qlearn(Node::DecisionTree; α::Float64=1.0, branch=Nothing())

    A = [Node.state.Q.A1, Node.state.Q.A1]

    a_idx = findall(x->x==findmax(A)[1], A)

    π = softMax(A[a_idx][1], A)

    if typeof(Node.A1) == Nothing || typeof(Node.A2) == Nothing

        if π > rand() && a_idx == 1 || π < rand() && a_idx == 2
            Node.state.Q.A1 = (1-α)*Node.state.Q.A1 + α*Node.state.R
        elseif π > rand() && a_idx == 2 || π < rand() && a_idx == 1
            Node.state.Q.A2 = (1-α)*Node.state.Q.A2 + α*Node.state.R
        end

    else
        if π > rand() && a_idx == 1 || π < rand() && a_idx == 2
            Node.A1 = Qlearn(Node.A1,branch=1,α=α)
        elseif π > rand() && a_idx == 2 || π < rand() && a_idx == 1
            Node.A2 = Qlearn(Node.A2,branch=0,α=α)
        end
        Q_ = findmax(A)
        Node.state.Q = (1-α)*Node.state.Q + α*Q_
    end

    return Node

end

function treePlotter(Node, plotArray ::Array{Float64,1})

    if typeof(Node.A1) == Nothing || typeof(Node.A2) == Nothing
        prepend!(plotArray, Node.state.Q)
    else
        plotArray = [treePlotter(Node.A1,plotArray), treePlotter(Node.A2,plotArray)]
        prepend!(plotArray, Node.state.Q)
    end

    return plotArray
end

StepTask = buildStepTask(prob_init,rewards)

#StepTask.state

for i=1:10
    MF_StepTask = Qlearn(StepTask,α=0.9)
end

MF_StepTask.state.Q
MF_StepTask.A1.state.Q
MF_StepTask.A2.state.Q
MF_StepTask.A1.A1.state.Q
MF_StepTask.A1.A2.state.Q
MF_StepTask.A2.A1.state.Q
MF_StepTask.A2.A2.state.Q

plotArray = Array{Float64,1}([0.0])

plotArray = treePlotter(MF_StepTask,plotArray)

plot(plotArray)
