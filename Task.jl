using Random
push!(LOAD_PATH, pwd())
using MA934
############### SET UP WORLD #########################################

moves = Dict{String,Array{Int64,1}}("left"=>[0,-1],"up"=>[-1,0],"right"=>[0,1],"down"=>[1,0])

maze = [Dict("LU"=> 0.0) Dict("MU"=> 100.0) Dict("RU"=> -100.0); Dict("LD"=> 0.0) Dict("MD"=> 0.0) Dict("RD"=> 0.0)]

reward = Dict( "LU"=> Dict("state"=>[1,1],"left"=>-10.0,"up"=>-10.0,"right"=>-1.0,"down"=>-1.0),
    "MU"=> Dict("state"=>[1,2],"left"=>100.0,"up"=>100.0,"right"=>100.0,"down"=>100.0),
    "RU"=> Dict("state"=>[1,3],"left"=>-100.0,"up"=>-100.0,"right"=>-100.0,"down"=>-100.0),
    "LD"=> Dict("state"=>[2,1],"left"=>-10.0,"up"=>-1.0,"right"=>-1.0,"down"=>-10.0),
    "MD"=> Dict("state"=>[2,2],"left"=>-1.0,"up"=>-10.0,"right"=>-1.0,"down"=>-10.0),
    "RD"=> Dict("state"=>[2,3],"left"=>-1.0,"up"=>-1.0,"right"=>-10.0,"down"=>-10.0))

########################################################################

function nextmove(reward, moves, Q, a::String, y::Int64, x::Int64, ϵ)
    y_=1
    x_=1

    s_idx = [y,x].+ moves[a]

    if s_idx[1] > 2 || s_idx[1] < 1
        s_idx = [y,x]
    elseif s_idx[2] > 3 || s_idx[2] < 1
        s_idx = [y,x]
    elseif s_idx == [1,2] && a == "up"
        s_idx = [y,x]
    end

    s_ = collect(keys(maze[s_idx[1],s_idx[2]]))
    y_=reward[s_[1]]["state"][1]
    x_=reward[s_[1]]["state"][2]

    Q_a = Dict(m => Q[m][y_,x_] for m in keys(moves))
    a = findmax(Q_a)[2]

    if rand() > ϵ
        a = collect(keys(moves))[rand(1:4)]
    end

    return a,s_[1],y_,x_
end

function Qlearn(reward,moves,Q,a,s,y::Int64,x::Int64)

    γ = 1       # Discount Const
    α = 1       # Confidence/Learning rate Const
    ϵ = 0.7       # Boldness/Exploration Const

    if s == "MU" || s == "RU"
        s = "RD"
        a = collect(keys(moves))[rand(1:4)]
        return Q,a,s
        exit
    else
        a_,s_,y_,x_ = nextmove(reward,moves,Q,a,y,x,ϵ)

        Q[a][y,x] = (1-α)*Q[a][y,x]+α*( (collect(values(maze[y_,x_]))[1]+reward[s][a])+γ*Q[a_][y_,x_] )
        a=a_
        s=s_
        x=x_
        y=y_
    end

    return Q,a,s
end

Q_init=Dict("left"=>zeros(size(maze)[1],size(maze)[2]), "up"=>zeros(size(maze)[1],size(maze)[2]), "right"=>zeros(size(maze)[1],size(maze)[2]), "down"=>zeros(size(maze)[1],size(maze)[2]))

Q = Q_init
s = "RD"
a = collect(keys(moves))[rand(1:4)]

count = 0
δ = 10

function converge(δ,max)

    global count

    if abs(δ) == 0.0 && count <= max
        count += 1
        δ = count
    else
        count = 0
    end

    return δ
end

while abs(δ) >= 0.01   #|| sum(sum(values(Q))) < 1200


    Q_old = deepcopy(Q)

    x=reward[s]["state"][2]
    y=reward[s]["state"][1]

    global Q,a,s = Qlearn(reward,moves,Q,a,s,y,x)

    #global δ = sum(sum(values(Q))) - sum(sum(values(Q_old)))

    global δ = converge(sum(sum(values(Q))) - sum(sum(values(Q_old))), 10)
    println(δ)

end

reward["RU"]["left"]

#### Schematic of maze ####
println(" ----------------------------------------------------")

println(" |      ",round(Q["up"][1,1], sigdigits=3),"      |                |                |")

println(" |  ",round(Q["left"][1,1], sigdigits=3),"    ",round(Q["right"][1,1], sigdigits=3),"  |     ",reward["MU"]["left"],"      |     ",reward["RU"]["left"],"     |")

println(" |      ",round(Q["down"][1,1], sigdigits=3),"      |                |                |")

println(" -----------------##################-----------------")

println(" |      ",round(Q["up"][2,1], sigdigits=3),"      |      ",round(Q["up"][2,2], sigdigits=3),"      |     ",round(Q["up"][2,3], sigdigits=3),"     |")

println(" |  ",round(Q["left"][2,1], sigdigits=3),"    ",round(Q["right"][2,1], sigdigits=3),"  |  ",round(Q["left"][2,2], sigdigits=3),"    ", round(Q["right"][2,2], sigdigits=3),"  |   ",round(Q["left"][2,3], sigdigits=3),"    ",round(Q["right"][2,3], sigdigits=3)," |")

println(" |      ",round(Q["down"][2,1], sigdigits=3),"      |      ",round(Q["down"][2,2], sigdigits=3),"      |       ",round(Q["down"][2,3], sigdigits=3),"     |")

println(" ----------------------------------------------------")
