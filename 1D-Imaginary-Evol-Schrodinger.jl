using FFTW
using LinearAlgebra
using Plots

δx = 10^(-4) #Grid spacing (Most practical + accurate spacing on my machine)
L = 30 #Interval Size
i = 10^4

mass = 1 #Particle mass
δt = 10^(-2) #Time step spacing
j = 10^2

n = -15 #Real line starting value
xGrid = LinRange(n, L+n, Int64(i)) #Creates 1D discrete line of real space
k = LinRange(-π/(L), π/(L), Int64(i)) #Creates 1D discrete line of momentum space

potentialArray = .05 .* (xGrid).^2 #Potential values at each point on the grid

function IC(xArray)
    yArray = (1/(sqrt(10)*π))^(1/2) .* exp.(-((xArray.+5).^2) ./ (2*sqrt(10))) #Initial Condition Wavefunction
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

energyArray = zeros(Int64(90*j)+1)
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
ψ2 = splitStepEvolution(5)
ψ3 = splitStepEvolution(10)
ψ4 = splitStepEvolution(20)
ψ5 = splitStepEvolution(60)
ψ6 = splitStepEvolution(90)


plot(
    plot(xGrid, normalization(real(IC(xGrid))), title = "Initial", label=false),
    plot(xGrid, (real(ψ1)), title = "1 Second", label=false),
    plot(xGrid, potentialArray, title="Potential Well", label=false),
    plot(xGrid, (real(ψ2)), title = "5 Seconds", label=false),
    plot(xGrid, (real(ψ3)), title = "10 Seconds", label=false),
    plot(xGrid, (real(ψ4)), title = "20 Seconds", label=false),
    plot(xGrid, (real(ψ5)), title = "60 Seconds", label=false),
    plot(xGrid, (real(ψ6)), title = "90 Seconds", label=false),
    plot(1:90*j+1, energyArray, title="Energy Evolution", xlabel="Time(s)", label=false)
)

plot(
    plot(k, abs.(normalization(fftshift(fft(IC(xGrid))))), title = "Initial", label=false),
    plot(k, abs.(fftshift(fft(ψ1))), title = "1 Second", label=false),
    plot(k, potentialArray, title="Potential Well", label=false),
    plot(k, abs.(fftshift(fft(ψ2))), title = "5 Seconds", label=false),
    plot(k, abs.(fftshift(fft(ψ3))), title = "10 Seconds", label=false),
    plot(k, abs.(fftshift(fft(ψ4))), title = "20 Seconds", label=false),
    plot(k, abs.(fftshift(fft(ψ5))), title = "60 Seconds", label=false),
    plot(k, abs.(fftshift(fft(ψ6))), title = "90 Seconds", label=false),
    plot(1:90*j+1, energyArray, title="Energy Evolution", xlabel="Time(s)")
)