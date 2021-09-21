using Flux, BSON, CUDA, Statistics, LsqFit
##
include("ising.jl")
##

struct PhysicalObservables
    m::Float64
    δm::Float64
    E::Float64
    δE::Float64
    χ::Float64
    δχ::Float64
    C::Float64
    δC::Float64
end

function PhysicalObservables(mag, ener, beta, L)
    function autocor_time(x, max)
        a = Array{Float64}(undef, max)
        for j in 1:max
            a[j] = cor(x[1:end-max], x[1+j:end-max+j])
        end
        return sum(a) 
    end
    
    n = size(mag)[1]
    m = mean(mag)
    E = mean(ener)
    τ_m = autocor_time(mag, 200)
    τ_E = autocor_time(ener, 200)
    δm = sqrt(1 + 2τ_m)*std(mag)
    δE = sqrt(1 + 2τ_E)*std(ener)
    χ = (beta*L^2)*var(mag)
    C = (beta^2/L^2)*var(ener)

    bootstrap_χ = Array{Float64}(undef, n)
    bootstrap_C = Array{Float64}(undef, n)
    trial_mag = Array{Float64}(undef, n)
    trial_ener = Array{Float64}(undef, n)
    for i in 1:n
        for j in 1:n
            trial_mag[j] = mag[rand(1:n)]
            trial_ener[j] = ener[rand(1:n)]
        end
        bootstrap_χ[i] = (beta*L^2)*var(trial_mag)
        bootstrap_C[i] = (beta^2/L^2)*var(trial_ener)
    end
    δχ = std(bootstrap_χ)
    δC = std(bootstrap_C)
    return PhysicalObservables(m, δm, E, δE,
            χ, δχ, C, δC)
end

function smart_mc(model, L, batch_size, beta; n=1, c=missing)
    
    function RegressionCoef() #Or, I can save these coefficients as BSON
        x = sample(model, L, 1000) |> cpu #along with the model
        lp = log_prob(model, x) |> cpu
        E = H(x, "square", false) |> cpu
        @. reg(e, p) = -p[1]*beta*e - p[2]
        fit = curve_fit(reg, E, lp, [1.,1000.])
        return coef(fit)
    end

    lp = Array{Float64}(undef, 0)
    E = Array{Float64}(undef, 0)
    m = Array{Float64}(undef, 0)
    if ismissing(c)
        c = RegressionCoef()
    end
    for i in 1:n
        x2 = sample(model, L, batch_size)
        lp2 = log_prob(model, x2) |> cpu
        e2 = H(x2, "square", false) |> cpu
        m2 = dropdims(mean(x2, dims=(1,2,3)), dims=(1,2,3)) |> cpu
        #Removing samples that are outlying the straight line fit
        ind = Tuple([j for j=1:batch_size if lp2[j]+beta*c[1]*e2[j]<-c[2]])
        m = cat(m, m2[setdiff(1:end, ind)], dims=1)
        lp = cat(lp, lp2[setdiff(1:end, ind)], dims=1)
        E = cat(E, e2[setdiff(1:end, ind)], dims=1)
    end

    ptr = 1
    acc = 0
    mag = Array{Float64}(undef, size(E)...)
    ener = Array{Float64}(undef, size(E)...)
    mag[1] = m[1]
    ener[1] = E[1]
    for j in 2:size(E)[1]
        arg = beta*(E[ptr]-E[j]) + (lp[ptr]-lp[j])
        if rand() < exp(arg)
            ptr = j
            acc += 1
        end
        mag[j] = abs(m[ptr])
        ener[j] = E[ptr]
    end
    @info "Acceptance Ratio" acc/(size(E)[1]-1.) size(E)[1]
    return mag, ener
end
##
BSON.@load "Saves/mish_linreg_(9,16,2).bson" cpu_model c
model = gpu(cpu_model)
##
mag, ener = smart_mc(model, 32, 800, 0.45, n=20, c=c)
##