using FFTW
using QuadGK
using Plots

δx = 10^(-6) #Grid spacing
t = 1 #Complete time evolution
δt = 10^(-4) #Time step spacing

function IC(xArray :: Vector{Float64})
    yArray = sqrt(2) .* sin.(π.*xArray) #Initial Condition Wavefunction
    return yArray #Numerical Array representing wavefunction
end

function kineticOperatorStep(m, ψ, xArray) #Time Step Operation for the Kinetic Operator
    kDomain = fft(ψ)
    N = length(ψ)
    k = (δx/N) .* xArray #Transforming the real line grid into the momentum space grid
    weightApplication = exp.((-k.^2)./(2*m) .* (δt / 2)) .* kDomain #Weighting each value in momentum space by the diagonalization factor
    xDomain = ifft(weightApplication)
    return xDomain
end

function potentialOperatorStep(V, ψ) #Time Step Operation for the Potential Operator
    operatedPsi = exp.(-V .* δt) .* ψ 
    return operatedPsi
end

function normalization(ψ) #Normalization of the Wavefunction (done discretely)
    absPsi = abs2.(ψ)
    L2Norm = sqrt(sum(absPsi .* δx)) #Discrete L2 calculation with grid space weighting
    return (ψ / L2Norm)
end



τ = δt #time step counter
i = 1/(δx) #Number of discrete values on the real line grid
mass = 1 #Particle mass

m = 0 #Real line starting value
xGrid = Array(m:δx:(m+((i-1)*δx))) #Creates 1D discrete line of real space
ϵ = 10^-20 #Convergence factor

potentialArray = zeros(Int64(i)) #Potential values at each point on the grid
ψ = IC(xGrid) #Initial condition

while τ <= t
    
    ψ0 = ψ

    #Split-Step Evolution
    global ψ = kineticOperatorStep(mass, ψ, xGrid)
    ψ = normalization(ψ)
    ψ = potentialOperatorStep(potentialArray, ψ)
    ψ = kineticOperatorStep(mass, ψ, xGrid)
    ψ = normalization(ψ)

    global τ += δt
end


plot(xGrid, real(ψ), label = "Re(ψ(x, t))", xlabel = "Position(x)", ylabel = "Wavefunction(ψ)")