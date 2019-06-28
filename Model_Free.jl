using Random, Plots, Pkg, TreeView

############### SET UP WORLD #########################################
push!(LOAD_PATH, pwd())
using CustomStructs

function buildStepTask(prob::Array{Float64,1},rewards::Array{Float64,1})

    if length(prob) == 1

        Task = DecisionTree(State(0.0,prob[1],rewards),Nothing(),Nothing())

    else

        Task = DecisionTree( State(0.0,prob[1],[0.0,0.0]), buildStepTask(prob[2:end],rewards), buildStepTask(prob[2:end],rewards) )

    end

    return Task
end

rewards = [1.0,0.0]

prob_init = [0.7,0.7,1.0]

StepTask = buildStepTask(prob_init,rewards)

#@tree buildStepTask(prob_init,rewards)

StepTask.state.Prob

########################################################################

function Qlearn(Node)

    γ = 0.9       # Discount Const
    α = 1       # Confidence/Learning rate Const
    #ϵ = 1       # Boldness/Exploration Const

    if typeof(Node.A1) == Nothing || typeof(Node.A2) == Nothing
        if Node.state.Prob > rand()
            Node.state.Q = (1-α)*Node.state.Q + α*Node.state.reward[1]
        else
            Node.state.Q = (1-α)*Node.state.Q + α*Node.state.reward[2]
        end

    else
        if Node.state.Prob > rand()
            Node.A1 = Qlearn(Node.A1)
        else
            Node.A2 = Qlearn(Node.A2)
        end

        Q_ = findmax([Node.A1.state.Q Node.A2.state.Q])[1]  #best next action

        #Q = (1-α)*Q+α*(r+γ*Q_))     #states and actions are wrapped up in tree strut

        Node.state.Q = (1-α)*Node.state.Q + α*γ*Q_
    end

    return Node

end

for i=1:500
    MF_StepTask = Qlearn(StepTask)
end

MF_StepTask

function treePlotter(Node,count)

    plotArray = Array{Float64,2}(UndefInitializer(),0,2)

    if typeof(Node.A1) == Nothing || typeof(Node.A2) == Nothing
        prepend!(plotArray, count,Node.state.Q)

    else
        plotter(Node.A1,count+1)
        plotter(Node.A2,count-1)

        prepend!(plotArray, count,Node.state.Q)
    end

    return plotArray
end

treePlotter(MF_StepTask,0)

scatter(treePlotter(MF_StepTask,0)[1,:],treePlotter(MF_StepTask,0)[2,:])
