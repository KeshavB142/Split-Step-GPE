using FFTW
using LinearAlgebra
using SparseArrays
using Plots

δx = 10^(-2) #Grid spacing (Most practical + accurate spacing on my machine)
L = 200 #Interval Size
i = Int64(round(1 / δx))

mass = 1 #Particle mass
δt = 10^(-2) #Time step spacing
j = Int64(round(1 / δt))

n = -100 #Real line starting value
xGrid = LinRange(n, L+n, Int64(i)) #Creates x-axis for the 3D space
yGrid = LinRange(n, L+n, Int64(i)) #Creates y-axis for the 3D space
Kx = LinRange(-π/(L), π/(L), Int64(i)) #Creates x-axis for the momentum space
Ky = LinRange(-π/(L), π/(L), Int64(i)) #Creates y-axis for the momentum space

kNormGrid = spzeros(Int64(i), Int64(i))
for i in eachindex(Kx)
    kNormGrid[i, :] = sqrt.(Kx[i]^2 .+ Ky.^2)
end
kNormGrid = Matrix(kNormGrid) #Norm of momentum space grid vectors

Kx = Ky = nothing #Removing momentum space building vectors from memory


potentialArray = spzeros(Int64(i), Int64(i))
for i in eachindex(xGrid)
    for j in eachindex(yGrid)
        potentialArray[i,j] = 0.005 .* (xGrid[i]^2 + yGrid[j]^2) #Potential values at each point on the grid
    end
end
potentialArray = Matrix(potentialArray)
N = 10^3 #Number of particles
g = 1.75 #Interaction strength



function IC(xArray, yArray)
    zArray = spzeros(Int64(i), Int64(i))
    for i in eachindex(xArray)
        zArray[i,:] = exp.((-(xArray[i] - 25).^2) ./ 1000) * exp.((-(yArray .+ 50).^2) ./ 1000) #Infinite Square Well Wavefunction
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
    X = -1 .* (V .+ g.*abs2.(ψ1)) .* δt
    operatedPsi = exp.(X) .* ψ1 
    return operatedPsi
end

function normalization(ψ1) #Normalization of the Wavefunction (done discretely)
    L2Norm = sqrt(sum(abs2.(ψ1) .* δx.^2)) #Discrete L2 calculation with grid space weighting
    return (ψ1 ./ L2Norm)
end

function calculatedChemPotential(ψ1)
    kψ = fftshift(fft(ψ1))
    ψstar = conj(ψ1)
    KE = (1/(2*mass)) * ψstar .* ifft(ifftshift((kNormGrid.^2) .* kψ))
    PE = ψstar .* potentialArray .* ψ1
    IE = g .* abs2.(ψ1)
    return sqrt(sum(abs2.(KE+PE+IE) .* δx))
end

μArray = zeros(Int64(10*j))
μArray[1] = calculatedChemPotential(sqrt(N) .* normalization(IC(xGrid,yGrid)))
function splitStepEvolution(time)

    t = time #Complete time evolution
    Nmax = Int64(round(t/(δt)))
    ψ = sqrt(N) .* normalization(IC(xGrid, yGrid)) #Initial condition Assignment
    #Split Step Evolution
    for i = 1:Nmax
        ψ = sqrt(N) .* normalization(kineticOperatorStep(mass,potentialOperatorStep(potentialArray, normalization(kineticOperatorStep(mass, ψ)))))
        μArray[i] = calculatedChemPotential(ψ)
    end
    return ψ
end

ψout= [splitStepEvolution(i) for i in 0:1:10] #Time period evolved Wavefns
timeLabelArray = ["t="*string(i)*"s" for i in 0:1:10] #Label array for time lines

plot(
    surface(xGrid, yGrid, real(ψout[1]), colorbar=false),
    surface(xGrid, yGrid, real(ψout[3]), colorbar=false),
    surface(xGrid, yGrid, real(ψout[5]), colorbar=false),
    surface(xGrid, yGrid, real(ψout[7]), colorbar=false),
    surface(xGrid, yGrid, real(ψout[9]), colorbar=false),
    surface(xGrid, yGrid, real(ψout[10]), colorbar=false),
    #surface(xGrid, yGrid, real(ψout[7] .* conj(ψout[7])), colorbar=false),
    title = "|ψ|^2 vs. Position"
)

#plot(1:60*j, μArray, title="μ Evolution", xlabel="Time(s)", label=false)
#.* conj(ψout[10])) <-- To calculate <ψ|ψ>