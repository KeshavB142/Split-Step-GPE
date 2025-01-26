using FFTW
using LinearAlgebra
using Plots

δx = 10^(-6) #Grid spacing (Most practical + accurate spacing on my machine)
L = 50 #Interval Size
i = 10^6

mass = 1 #Particle mass
δt = 10^(-2) #Time step spacing
j = 10^2

n = -25 #Real line starting value
xGrid = LinRange(n, L+n, Int64(i)) #Creates 1D discrete line of real space
k = LinRange(-π/(L), π/(L), Int64(i)) #Creates 1D discrete line of momentum space

potentialArray = .05 .* (xGrid).^2 #Potential values at each point on the grid

function IC(xArray)
    yArray = (1/(sqrt(10)*π))^(1/2) .* exp.(-((xArray .+ 10).^2) ./ (2*sqrt(10))) #Initial Condition Wavefunction
    return yArray #Numerical Array representing wavefunction
end

function kineticOperatorStep(m, ψ1) #Time Step Operation for the Kinetic Operator 
    weights = exp.(-1*((k.^2)./(2*m)) .* (δt / 2)) #Calculation of the diagonalization factor
    weightedψ = weights .* (fft(ψ1))  #Weighting each value in momentum space by the diagonalization factor
    xDomain = ifft(weightedψ)
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
    kψ = fft(ψ1)
    ψstar = conj(ψ1)
    KE = (1/(2*mass)) * ψstar .* ifft((k.^2) .* kψ)
    PE = ψstar .* potentialArray .* ψ1
    return sqrt(sum(abs2.(KE+PE) .* δx))
end

energyArray = zeros(Int64(20*j)+1)
energyArray[1] = calculatedEnergy(normalization(IC(xGrid)))
function splitStepEvolution(time)

    t = time #Complete time evolution

    τ = δt #time step counter
    ϵ = 10^-20 #Convergence factor

    ψ = IC(xGrid) #Initial condition Assignment
    i=1
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
        i+=1
        energyArray[i] = calculatedEnergy(ψ)

    end
    return ψ
end

ψ1 = splitStepEvolution(1)
ψ2 = splitStepEvolution(10)
ψ3 = splitStepEvolution(5)
ψ4 = splitStepEvolution(20)

plot(
    plot(xGrid, normalization(real(IC(xGrid))), label = "Initial"),
    plot(xGrid, real(ψ1), label = "1 Second", xlabel = "Position(x)", ylabel = "Wavefunction(ψ)"),
    plot(xGrid, potentialArray, label="Potential Well"),
    plot(xGrid, real(ψ2), label = "5 Seconds"),
    plot(xGrid, real(ψ3), label = "10 Seconds"),
    plot(xGrid, real(ψ4), label = "20 Seconds"),
    plot(1:j+1, energyArray, label="Energy Evolution", xlabel="Time(s)")
)