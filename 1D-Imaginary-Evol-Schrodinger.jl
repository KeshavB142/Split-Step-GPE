using FFTW
using LinearAlgebra
using Plots

δx = 10^(-3) #Grid spacing (Most practical + accurate spacing on my machine)
L = 30 #Interval Size
i = Int64(round(1 / δx))

mass = 1 #Particle mass
δt = 10^(-2) #Time step spacing
j = Int64(round(1 / δt))

n = -15 #Real line starting value
xGrid = LinRange(n, L+n, i) #Creates 1D discrete line of real space
k = LinRange(-π/(L), π/(L), i) #Creates 1D discrete line of momentum space

potentialArray = .005 .* ((xGrid).^2) #Potential values at each point on the grid

function IC(xArray)
    yArray = exp.((-(xArray .- 3).^2)) #Initial Condition Wavefunction
    return yArray #Numerical Array representing wavefunction
end

function kineticOperatorStep(m, ψ1) #Time Step Operation for the Kinetic Operator 
    weights = exp.(-1*((k.^2)./(2*m)) .* (δt / 2)) #Calculation of the diagonalization factor
    weightedψ = weights .* (fftshift(fft(ψ1)))  #Weighting each value in momentum space by the diagonalization factor
    xDomain = ifft(ifftshift(weightedψ))
    return xDomain
end

function potentialOperatorStep(V, ψ1) #Time Step Operation for the Potential Operator
    X = -1 .* V .* δt
    operatedPsi = exp.(X) .* ψ1 
    return operatedPsi
end

function normalization(ψ1) #Normalization of the Wavefunction (done discretely)
    L2Norm = sqrt(sum(abs2.(ψ1) .* δx)) #Discrete L2 calculation with grid space weighting
    return (ψ1 ./ L2Norm)
end

function calculatedEnergy(ψ1)
    kψ = fftshift(fft(ψ1))
    ψstar = conj(ψ1)
    KE = (1/(2*mass)) * ψstar .* ifft(ifftshift((k.^2) .* kψ))
    PE = ψstar .* potentialArray .* ψ1
    return sqrt(sum(abs2.(KE+PE) .* δx))
end

energyArray = zeros((360*j)+1)
energyArray[1] = calculatedEnergy(normalization(IC(xGrid)))
function splitStepEvolution(time)

    t = time #Complete time evolution
    Nmax = Int64(round(t/(δt)))
    ψ = normalization(IC(xGrid)) #Initial condition Assignment
    #Split Step Evolution
    for i = 1:Nmax
        ψ = normalization(kineticOperatorStep(mass,potentialOperatorStep(potentialArray, normalization(kineticOperatorStep(mass, ψ)))))
        energyArray[i] = calculatedEnergy(ψ)
    end
    return ψ
end

ψout= [splitStepEvolution(i) for i in 0:45:360] #Time period evolved Wavefns
timeLabelArray = ["t="*string(i)*"s" for i in 0:45:360] #Label array for time lines

plot(xGrid, potentialArray, title = "Wavefunction vs. Position", label="Potential Well")
plot!(xGrid, real(ψout))

#label = timeLabelArray[i]
#plot(1:5*j+1, energyArray, title="Energy Evolution", xlabel="Time(s)", label=false)