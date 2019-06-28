using Plots

# Bellmann
function Bellmann(s,V,r,P,moves)

    V[s] = r[s] + γ*sum(P[s for s in moves]*V[s for s in moves])

end
# Value Iteration
function VI()

    poss_moves[a] = [moves[a,1] moves[a,2] moves[a,3] moves[a,4]]

    # Bellman(s,V,r,P,poss_moves)
    V[s] = r[s] + γ*sum(P[s for s in moves]*V[s for s in moves])

end

# Policy Iteration


# SARSA


# Temporal-Difference


# Q

γ = 1       # Discount Const
α = 1       # Confidence/Learning rate Const
ϵ = 1       # Boldness/Exploration Const
