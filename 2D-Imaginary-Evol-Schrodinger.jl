using FFTW
using LinearAlgebra
using SparseArrays
using Plots

δx = 10^(-2) #Grid spacing (Most practical + accurate spacing on my machine)
L = 1 #Interval Size
i = 10^2

mass = 1 #Particle mass
δt = 10^(-2) #Time step spacing
j = 10^2

n = 0 #Real line starting value
xGrid = LinRange(n, L+n, Int64(i)) #Creates x-axis for the 3D space
yGrid = LinRange(n, L+n, Int64(i)) #Creates y-axis for the 3D space
Kx = LinRange(-π/(L), π/(L), Int64(i)) #Creates x-axis for the momentum space
Ky = LinRange(-π/(L), π/(L), Int64(i)) #Creates y-axis for the momentum space

potentialArray = spzeros(Int64(i), Int64(i))
#for i in eachindex(xGrid)
#    for j in eachindex(yGrid)
#        potentialArray[i,j] = 0.5 .* (xGrid[i]^2 + yGrid[j]^2) #Potential values at each point on the grid
#    end
#end
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
        zArray[i,:] = exp.(-xArray[i].^2) * exp.(-yArray.^2) #Infinite Square Well Wavefunction
    end #Initial Condition Wavefunction
    return Matrix(zArray) #Numerical Array representing wavefunction
end

function kineticOperatorStep(m, ψ1) #Time Step Operation for the Kinetic Operator 
    weights = exp.(-1*((kNormGrid.^2)./(2*m)) .* (δt / 2)) #Calculation of the diagonalization factor
    weightedψ = weights .* fftshift(fft(ψ1))  #Weighting each value in momentum space by the diagonalization factor
    xDomain = ifft(ifftshift(weightedψ))
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

function calculatedEnergy(ψ1)
    kψ = fftshift(fft(ψ1))
    ψstar = conj(ψ1)
    KE = (1/(2*mass)) * ψstar .* ifft(ifftshift((kNormGrid.^2) .* kψ))
    PE = ψstar .* potentialArray .* ψ1
    return sqrt(sum(abs2.(KE+PE) .* δx))
end

energyArray = zeros((360*j)+1)
energyArray[1] = calculatedEnergy(normalization(IC(xGrid,yGrid)))
function splitStepEvolution(time)

    t = time #Complete time evolution
    Nmax = Int64(round(t/(δt)))
    ψ = normalization(IC(xGrid, yGrid)) #Initial condition Assignment
    #Split Step Evolution
    for i = 1:Nmax
        ψ = normalization(kineticOperatorStep(mass,potentialOperatorStep(potentialArray, normalization(kineticOperatorStep(mass, ψ)))))
        energyArray[i] = calculatedEnergy(ψ)
    end
    return ψ
end

ψout= [splitStepEvolution(i) for i in 0:72:360] #Time period evolved Wavefns
timeLabelArray = ["t="*string(i)*"s" for i in 0:72:350] #Label array for time lines

plot(
    #surface(xGrid, yGrid, potentialArray, title = "Potential Well", colorbar=false),
    surface(xGrid, yGrid, real(ψout[1]), colorbar=false),
    surface(xGrid, yGrid, real(ψout[2]), colorbar=false),
    surface(xGrid, yGrid, real(ψout[3]), colorbar=false),
    surface(xGrid, yGrid, real(ψout[4]), colorbar=false),
    surface(xGrid, yGrid, real(ψout[5]), colorbar=false),
    surface(xGrid, yGrid, real(ψout[6]), colorbar=false),
)
#label = timeLabelArray[i]
#plot(1:5*j+1, energyArray, title="Energy Evolution", xlabel="Time(s)", label=false)
