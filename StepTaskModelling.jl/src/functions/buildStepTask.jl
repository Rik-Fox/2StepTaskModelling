
## function that buils a tree 'object' with module defined data types 
function buildStepTask(steps::Int;TM::Bool=false,r::Float64=0.0)
    if steps == 1
        Q = Actions(0.0,0.0)
        TM ? T = Actions(0.5,0.5) : T = Nothing()
        h = Actions(0.0,0.0)
        R = 0.0
        Task = DecisionTree(State(Q,T,h,R),round(0.8-r,digits=2),0.0)
    else
        Q = Actions(0.0,0.0)
        TM ? T = Actions(0.5,0.5) : T = Nothing()
        h = Actions(0.0,0.0)
        R = 0.0
        Task = DecisionTree(State(Q,T,h,R), buildStepTask(steps-1,TM=TM), buildStepTask(steps-1,TM=TM,r=0.6))
    end

    return Task
end
