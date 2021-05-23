# Simulation of Ising model with Monte Carlo, Metropolis Algorithm

using Random.Random
using Plots

function twoD(Field, E, M, C, X, B, Q, N, relax_time, nT, epochs)
    N2 = N*N
    iMCs = 1.0/epochs;
    iNs = 1.0/N2;

    initialise(Field, N)

    for tt in 1:1:nT
        println(tt, " ", nT)
        E1 = E2 = M1 = M2 = 0;
        Mag = 0
        Ene = 0
        beta = B[tt]

        for i in 1:1:relax_time
            ising_step(Field, beta, N)
        end
        for j in 1:1:epochs
            ising_step(Field, beta, N)

            Ene = calcEne(Field, N)
            Mag = calcMag(Field, N)
            E1 = E1 + Ene;
            M1 = M1 + Mag;
            E2 = E2 + Ene*Ene;
            M2 = M2 + Mag*Mag;
        end
        E[tt] = E1*iMCs*iNs
        M[tt] = M1*iMCs*iNs
        C[tt] = (E2* - E1*E1*iMCs*iMCs)*beta*beta*iNs;
        X[tt] = (M2*iMCs - M1*M1*iMCs*iMCs)*beta*iNs;
        Q[tt] = M2*iMCs/(M1*M1*iMCs*iMCs)
    end
    return
end

function initialise(Field, N)
    for i in 1:1:N+2
        for j in 1:1:N+2
            if rand(Float64) < 0.5
                Field[i,j] = -1
            else
                Field[i,j] = 1
            end
        end
    end
    return 0
end


function ising_step(Field, beta, N)
    for ii in 1:1:(N*N)
        a = 1 + rand(1:N)
        b = 1 + rand(1:N)

        Field[1, b]   = Field[N+1, b];
        Field[N+2, b] = Field[2, b];  # ensuring BC
        Field[a, 1]   = Field[a, N+1];
        Field[a, N+2] = Field[a, 2];

        dE = 2*Field[a,b]*(Field[a+1, b] + Field[a,b+1] + Field[a-1, b] + Field[a,b-1])
        if dE < 0
            Field[a,b] *= -1
        elseif rand() < exp(-dE*beta)
            Field[a,b] *= -1
        end
    end
    return 0

end

function calcEne(Field, N)
    energy = 0
    for i in 2:1:(N+1)
        for j in 2:1:(N+1)
            Field[1, j] = Field[N, j];
            Field[i, 1] = Field[i, N];
            energy += -Field[i, j] * (Field[i-1, j] + Field[i, j-1])
        end
    end
    return energy/2

end

function calcMag(Field, N)
    mag = 0
    for i in 1:1:(N+1)
        for j in 1:1:(N+1)
            mag += Field[i,j]
        end
    end
    return mag

end


# Execution

# model parameters
N = 32
nT = 30
epochs = 4000
relax_time = 1000

# observables arrays
Field           = zeros(Int32, N+2, N+2)
Energy          = zeros(Float64, nT)
Magnetization   = zeros(Float64, nT)
SpecificHeat    = zeros(Float64, nT)
Susceptibility  = zeros(Float64, nT)
Binder          = zeros(Float64, nT)
Error_Ene       = zeros(Float64, nT)
Error_Mag       = zeros(Float64, nT)

Temperature = range(1.0, stop = 4.0, length = nT)
Beta = 1.0/Temperature

twoD(Field, Energy, Magnetization, SpecificHeat, Susceptibility,
           Beta, Binder, N, relax_time, nT, epochs)


mag_plot = plot(Temperature, abs.(Magnetization))
#ene_plot = plot(Temperature, Energy)
#chi_plot = plot(Temperature, Susceptibility)
#heat_plot = plot(Temperature, SpecificHeat)
