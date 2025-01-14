using FFTW
using QuadGK
using Plots

δx = 10^(-6) #Grid spacing (Most practical + accurate spacing on my machine)
L = 1 #Interval Size
i = 10^6

t = 1 #Complete time evolution
δt = 10^(-2) #Time step spacing

mass = 1 #Particle mass

n = 0 #Real line starting value
xGrid = LinRange(n, L, Int64(i)) #Creates 1D discrete line of real space
k = π/(L) .* vcat(0:Int64((i))/2 - 1, -Int64((i))/2:-1) #Creates 1D discrete line of momentum space

potentialArray = zeros(Int64(i)) #Potential values at each point on the grid

function IC(xArray)
    yArray = sqrt(2) .* sin.(π.* xArray) #Initial Condition Wavefunction
    return yArray #Numerical Array representing wavefunction
end

function kineticOperatorStep(m, ψ1) #Time Step Operation for the Kinetic Operator 
    weights = exp.(-1*((k.^2)./(2*m)) .* (δt / 2)) #Calculation of the diagonalization factor
    weightedψ = weights .* fft(ψ1)  #Weighting each value in momentum space by the diagonalization factor
    xDomain = ifft(weightedψ)
    return xDomain
end

function potentialOperatorStep(V, ψ1) #Time Step Operation for the Potential Operator
    operatedPsi = exp.(V .* δt ) .* ψ1 
    return operatedPsi
end

function normalization(ψ1) #Normalization of the Wavefunction (done discretely)
    L2Norm = sqrt(sum(abs2.(ψ1) .* δx)) #Discrete L2 calculation with grid space weighting
    return (ψ1 ./ L2Norm)
end



τ = δt #time step counter
ϵ = 10^-20 #Convergence factor

ψ = IC(xGrid) #Initial condition Assignment

while τ <= t

    #Split-Step Evolution
    global ψ = kineticOperatorStep(mass, ψ)
    #print(maximum(real(ψ)) )
    ψ = normalization(ψ)
    #print(maximum(real(ψ)))
    ψ = potentialOperatorStep(potentialArray, ψ)
    #print(maximum(real(ψ)))
    ψ = kineticOperatorStep(mass, ψ)
    #print(maximum(real(ψ)))
    ψ = normalization(ψ)
    println(maximum(real(ψ)))

    global τ += δt
end


plot(xGrid, real(ψ), label = "Re(ψ(x, t))", xlabel = "Position(x)", ylabel = "Wavefunction(ψ)")
ylims!(minimum(real(ψ)), maximum(real(ψ)))