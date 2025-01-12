using FFTW
using QuadGK

m = 1 #Real line starting value
δx = 10^(-5) #Grid spacing
t = 1 #Complete time evolution
δt = 1/(10^-10) #Time step spacing
function IC(n)
    yArray = ones(n)
    while m <= n*δx #Iterate until the max of the array length is reached with appropriate input values
        yArray[m] = exp(-m) #Initial Condition Wavefunction
        global m += δx
    end
    return yArray #Numerical Array representing wavefunction
end

function kineticOperatorStep(m, ψ, xArray) #Time Step Operation for the Kinetic Operator
    kDomain = fft(ψ)
    N = length(ψ)
    k = (δx/N) .* xArray #Transforming the real line grid into the momentum space grid
    weightApplication = exp.((-k.^2)./(2m) .* (δt / 2)) .* kDomain #Weighting each value in momentum space by the diagonalization factor
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

potentialArray = zeros(n) #Potential values at each point on the grid
τ = δt #time step counter
i = 1000 #Number of discrete values on the real line grid
ψ = IC(i) #Initial condition
mass = 9.1*(10^-31) #Particle mass
xGrid = Array(0:δx:((i-1)*δx)) #Creates 1D discrete line of real space
ϵ = 10^-15 #Convergence factor
while τ <= t
    
    ψ0 = ψ

    #Split-Step Evolution
    ψ = kineticOperatorStep(mass, ψ, xGrid)
    ψ = normalization(ψ)
    ψ = potentialOperatorStep(potentialArray, ψ)
    ψ = kineticOperatorStep(mass, ψ, xGrid)
    ψ = normalization(ψ)

    if max(ψ - ψ0) < ϵ*δt
        break
    end
    τ += δt
end

