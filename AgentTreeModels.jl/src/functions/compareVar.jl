function compareVar(agent::DecisionTree, C::DecisionTree, TV::DecisionTree, epochC::Array{Float64,2}, epochTV::Array{Float64,2})


    Cσ = (sum(epochC.^2, dims = 2)./length(epochC[:,end])) .- (sum(epochC, dims = 2) ./ length(epochC[:,end])).^2
    TVσ = (sum(epochTV.^2, dims = 2)./length(epochC[:,end])) .- (sum(epochTV, dims = 2) ./ length(epochC[:,end])).^2

    if TVσ[1] < Cσ[1]
        agent.state.Q.A1 = deepcopy(TV.state.Q.A1)
        agent.state.e.A1 = deepcopy(TV.state.e.A1)
    else
        agent.state.Q.A1 = deepcopy(C.state.Q.A1)
        agent.state.e.A1 = deepcopy(C.state.e.A1)
    end

    if TVσ[2] < Cσ[2]
        agent.state.Q.A2 = deepcopy(TV.state.Q.A2)
        agent.state.e.A2 = deepcopy(TV.state.e.A2)
    else
        agent.state.Q.A2 = deepcopy(C.state.Q.A2)
        agent.state.e.A2 = deepcopy(C.state.e.A2)
    end

    if TVσ[3] < Cσ[3]
        agent.μ.state.Q.A1 = deepcopy(TV.μ.state.Q.A1)
        agent.μ.state.e.A1 = deepcopy(TV.μ.state.e.A1)
    else
        agent.μ.state.Q.A1 = deepcopy(C.μ.state.Q.A1)
        agent.μ.state.e.A1 = deepcopy(C.μ.state.e.A1)
    end

    if TVσ[4] < Cσ[4]
        agent.μ.state.Q.A2 = deepcopy(TV.μ.state.Q.A2)
        agent.μ.state.e.A2 = deepcopy(TV.μ.state.e.A2)
    else
        agent.μ.state.Q.A2 = deepcopy(C.μ.state.Q.A2)
        agent.μ.state.e.A2 = deepcopy(C.μ.state.e.A2)
    end

    if TVσ[5] < Cσ[5]
        agent.ν.state.Q.A1 = deepcopy(TV.ν.state.Q.A1)
        agent.ν.state.e.A1 = deepcopy(TV.ν.state.e.A1)
    else
        agent.ν.state.Q.A1 = deepcopy(C.ν.state.Q.A1)
        agent.ν.state.e.A1 = deepcopy(C.ν.state.e.A1)
    end

    if TVσ[6] < Cσ[6]
        agent.ν.state.Q.A2 = deepcopy(TV.ν.state.Q.A2)
        agent.ν.state.e.A2 = deepcopy(TV.ν.state.e.A2)
    else
        agent.ν.state.Q.A2 = deepcopy(C.ν.state.Q.A2)
        agent.ν.state.e.A2 = deepcopy(C.ν.state.e.A2)
    end

    return agent
end
