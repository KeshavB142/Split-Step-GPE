using FFTW
using LinearAlgebra
using SparseArrays
using Plots

δx = 10^(-2) #Grid spacing (Most practical + accurate spacing on my machine)
L = 20 #Interval Size
i = 10^2

mass = 1 #Particle mass
δt = 10^(-2) #Time step spacing

n = -10 #Real line starting value
xGrid = LinRange(n, L+n, Int64(i)) #Creates x-axis for the 3D space
yGrid = LinRange(n, L+n, Int64(i)) #Creates y-axis for the 3D space
Kx = LinRange(-π/(L), π/(L), Int64(i)) #Creates x-axis for the momentum space
Ky = LinRange(-π/(L), π/(L), Int64(i)) #Creates y-axis for the momentum space

potentialArray = spzeros(Int64(i), Int64(i))
for i in eachindex(xGrid)
    for j in eachindex(yGrid)
        potentialArray[i,j] = 0.5 .* (xGrid[i]^2 + yGrid[j]^2) #Potential values at each point on the grid
    end
end
potentialArray = Matrix(potentialArray)

kNormGrid = spzeros(Int64(i), Int64(i))
for i in eachindex(Kx)
    kNormGrid[i, :] = sqrt.(Kx[i]^2 .+ Ky.^2)
end
kNormGrid = Matrix(kNormGrid) #Norm of momentum space grid vectors

Kx = Ky = nothing #Removing momentum space building vectors from memory

function IC(xArray, yArray)
    zArray = spzeros(Int64(i), Int64(i))
    for i in eachindex(xArray)
        for j in eachindex(yArray)
            zArray[i,j] = (1/(π)) .* exp(-1 * ((xArray[i]-4.0)^2) / 2) .* exp.(-1*((yArray[j]-2.0)^2)/ 2) #Infinite Square Well Wavefunction
        end
    end #Initial Condition Wavefunction
    return Matrix(zArray) #Numerical Array representing wavefunction
end

function kineticOperatorStep(m, ψ1) #Time Step Operation for the Kinetic Operator 
    weights = exp.(-1*((kNormGrid.^2)./(2*m)) .* (δt / 2)) #Calculation of the diagonalization factor
    weightedψ = weights .* fft(ψ1)  #Weighting each value in momentum space by the diagonalization factor
    xDomain = ifft(weightedψ)
    return xDomain
end

function potentialOperatorStep(V, ψ1) #Time Step Operation for the Potential Operator
    X = -1 .* V .* δt
    operatedPsi = exp.(X) .* ψ1 
    return operatedPsi
end

function normalization(ψ1) #Normalization of the Wavefunction (done discretely)
    L2Norm = sqrt(sum(abs2.(ψ1) .* δx.^2)) #Discrete L2 calculation with grid space weighting
    return (ψ1 ./ L2Norm)
end

function splitStepEvolution(time)

    t = time #Complete time evolution

    τ = δt #time step counter
    ϵ = 10^-20 #Convergence factor

    ψ = IC(xGrid, yGrid) #Initial condition Assignment

    while τ <= t

        #Split-Step Evolution
        ψ = kineticOperatorStep(mass, ψ)
        #print(maximum(real(ψ)) )
        ψ = normalization(ψ)
        #print(maximum(real(ψ)))
        ψ = potentialOperatorStep(potentialArray, ψ)
        #print(maximum(real(ψ)))
        ψ = kineticOperatorStep(mass, ψ)
        #print(maximum(real(ψ)))
        ψ = normalization(ψ)
        #println(maximum(real(ψ)))

        τ += δt
    end
    return ψ
end

ψ1 = splitStepEvolution(1)
ψ2 = splitStepEvolution(3)
ψ3 = splitStepEvolution(5)

plot(
    surface(xGrid, yGrid, potentialArray, title="Potential Well", colorbar=false),
    surface(xGrid, yGrid, normalization(real(IC(xGrid, yGrid))), title = "Initial Wavefunction", colorbar=false),
    surface(xGrid, yGrid, real(ψ1), title = "1 Second", xlabel = "Position(x)", ylabel = "Position (y)", zlabel = "Wavefunction(ψ)", colorbar=false),
    surface(xGrid, yGrid, real(ψ2), title = "3 Seconds", colorbar=false),
    surface(xGrid, yGrid, real(ψ3), title = "5 Seconds", colorbar=false),
#xlims!(xGrid[250000],xGrid[750000])
)