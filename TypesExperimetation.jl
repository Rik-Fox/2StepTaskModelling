using Revise

push!(LOAD_PATH, pwd())

using MyTypes

agent = buildAgent(2)

typeof(Actions)
typeof(State)
typeof(DecisionTree)

Float64 <: Actions

Actions <: State

Actions{Float64} <: State{Float64}
