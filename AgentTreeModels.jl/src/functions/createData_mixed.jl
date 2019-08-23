### Dezfouli

function createData_mixed(agent::DecisionTree, w::T, ϵ_cut::T; N::Int=100, α::T=0.5, β::T=5.0, λ::T=0.0, ηₜ::T=0.5, ηᵣ::T=0.5) where T<:Float64
    dat = Array{Any,2}(undef,(N,4))
    P𝑮 = ones(N)
    r𝑮 = 0.0
    r0 = 0.0
    h_avg = 0.0

    for i=1:N
        P𝑮[i] = 1/( 1 + exp( (w*abs(r𝑮 - r0)) + ((1-w)*abs(h_avg^2)) ) )

        if rand() < P𝑮[i]
            x = agentCtrller_mixed(agent, ϵ_cut, α=α, β=β, λ=λ, ηₜ=ηₜ, ηᵣ=ηᵣ)
            agent = x[1]
            dat[i,:] = [x[3][1],x[3][2],x[3][3],x[2]]
            r𝑮 = (1-α)*r𝑮 + α*x[2]
        else
            x = agentCtrller_mixed(agent, α=α, β=β, ηₜ=ηₜ, ηᵣ=ηᵣ)
            agent = x[1]
            dat[i,:] = [x[3][1],x[3][2],x[3][3],x[2]]
            h_avg = sum([agent.state.Q.A1, agent.state.Q.A2, agent.μ.state.Q.A1, agent.μ.state.Q.A2, agent.ν.state.Q.A1, agent.ν.state.Q.A2])/6
        end
        r0 = (1-α)*r0 + α*x[2]
    end
    Data = [[d=="A1" for d in dat[:,1]] [d for d in dat[:,2]] [d=="A1" for d in dat[:,3]] [d==1 for d in dat[:,4]]]

    return Data
end


##Miller

function createData_mixed(agent::DecisionTree, wₒ::T, wₕ::T, wᵧ::T, ϵ_cut::T; N::Int=100, α::T=0.5, β::T=5.0, λ::T=0.0, ηₜ::T=0.5, ηᵣ::T=0.5) where T<:Float64
    dat = Array{Any,2}(undef,(N,4))
    Pᵧ = ones(N)
    rᵧ = 0.0
    rₒ = 0.0
    H = 0.0

    for i=1:N
        Pᵧ[i] = 1/( 1 + exp( (wᵧ*abs(rᵧ - rₒ)) + (wₕ*abs(H^2) + wₒ) ) )

        if rand() < Pᵧ[i]
            x = agentCtrller_mixed(agent, ϵ_cut, α=α, β=β, λ=λ, ηₜ=ηₜ, ηᵣ=ηᵣ)
            agent = x[1]
            dat[i,:] = [x[3][1],x[3][2],x[3][3],x[2]]
            rᵧ = (1-α)*rᵧ + α*x[2]
        else
            x = agentCtrller_mixed(agent, α=α, β=β, ηₜ=ηₜ, ηᵣ=ηᵣ)
            agent = x[1]
            dat[i,:] = [x[3][1],x[3][2],x[3][3],x[2]]
            H = sum([agent.state.Q.A1, agent.state.Q.A2, agent.μ.state.Q.A1, agent.μ.state.Q.A2, agent.ν.state.Q.A1, agent.ν.state.Q.A2])/6
        end
        rₒ = (1-α)*rₒ + α*x[2]
    end
    Data = [[d=="A1" for d in dat[:,1]] [d for d in dat[:,2]] [d=="A1" for d in dat[:,3]] [d==1 for d in dat[:,4]]]

    return Data
end
