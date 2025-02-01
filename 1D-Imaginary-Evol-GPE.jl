using FFTW
using LinearAlgebra
using Plots

δx = 10^(-3) #Grid spacing (Most practical + accurate spacing on my machine)
L = 200 #Interval Size
i = Int64(round(1 / δx))

mass = 1 #Particle mass
δt = 10^(-2) #Time step spacing
j = Int64(round(1 / δt))

n = -100 #Real line starting value
xGrid = LinRange(n, L+n, i) #Creates 1D discrete line of real space
k = LinRange(-π/(L), π/(L), i) #Creates 1D discrete line of momentum space

potentialArray = 1 .* ((xGrid).^2) #Potential values at each point on the grid
N = 10^3 #Number of particles
g = 1.5 #Strength of interaction term

function IC(xArray)
    yArray = exp.((-(xArray .- 0).^2)/20) #Initial Condition Wavefunction
    return yArray #Numerical Array representing wavefunction
end

function kineticOperatorStep(m, ψ1) #Time Step Operation for the Kinetic Operator 
    weights = exp.(-1*((k.^2)./(2*m)) .* (δt / 2)) #Calculation of the diagonalization factor
    weightedψ = weights .* (fftshift(fft(ψ1)))  #Weighting each value in momentum space by the diagonalization factor
    xDomain = ifft(ifftshift(weightedψ))
    return xDomain
end

function potentialOperatorStep(V, ψ1) #Time Step Operation for the Potential Operator
    X = -1 .* (V .+ g.*abs2.(ψ1)) .* δt
    operatedPsi = (exp.(X)) .* ψ1 
    return operatedPsi
end

function normalization(ψ1) #Normalization of the Wavefunction (done discretely)
    L2Norm = sqrt(sum(abs2.(ψ1) .* δx)) #Discrete L2 calculation with grid space weighting
    return (ψ1 ./ L2Norm)
end

function calculatedChemPotential(ψ1)
    kψ = fftshift(fft(ψ1))
    ψstar = conj(ψ1)
    KE = (1/(2*mass)) * ψstar .* ifft(ifftshift((k.^2) .* kψ))
    PE = ψstar .* potentialArray .* ψ1
    IE = g .* abs2.(ψ1)
    return sqrt(sum(abs2.(KE+PE+IE) * δx))
end

μArray = zeros((60*j))
μArray[1] = calculatedChemPotential(sqrt(N) .* normalization(IC(xGrid)))
function splitStepEvolution(time)

    t = time #Complete time evolution
    Nmax = Int64(round(t/(δt)))
    ψ =  sqrt(N).*normalization(IC(xGrid)) #Initial condition Assignment
    #Split Step Evolution
    for i = 1:Nmax
        ψ = sqrt(N).*normalization(kineticOperatorStep(mass,potentialOperatorStep(potentialArray, normalization(kineticOperatorStep(mass, ψ)))))
        μArray[i] = calculatedChemPotential(ψ)
    end
    return ψ
end

ψout= [splitStepEvolution(i) for i in 0:3:60] #Time period evolved Wavefns
timeLabelArray = ["t="*string(i)*"s" for i in 0:3:60] #Label array for time lines

plt = plot()
plot!(plt, xGrid, potentialArray, title = "|ψ|^2 vs. Position", label="Potential Well")
for i = 1:21
    local mag = (ψout[i]) .* conj((ψout[i]))
    plot!(plt, xGrid, real(mag), label = string(timeLabelArray[i]))
end
current()

#, label=string(timeLabelArray[i])
#plot(1:15*j, μArray, title="μ Evolution", xlabel="Time(s)", label=false)