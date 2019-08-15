using Revise

push!(LOAD_PATH, pwd())

using MyTypes

agent = buildAgent(2,Qvalue=false,habit=true)

Ctrl(agent)

agent.state.Q.A1 = 6
agent.state.R = 12
agent
typeof(Actions)
typeof(State)
typeof(DecisionTree)


a = Nothing()

c=[1 2 3 4]

c

d=c[3:4]

c == d
