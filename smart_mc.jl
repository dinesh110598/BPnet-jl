using Flux, BSON, CUDA
##
include("ising.jl")
##
function smart_mc(model, L, batch_size, beta; n=1)
    x = Array{Float64}(undef, L, L, 1, 0)
    lp = Array{Float64}(undef, 0)
    E = Array{Float64}(undef, 0)
    for i in 1:n
        x2 = sample(model, L, batch_size)
        lp2 = log_prob(model, x2) |> cpu
        e2 = H(x2, "square", false) |> cpu
        x2 = cpu(x2)
        ind = Tuple([j for j=1:batch_size if lp2[j]+0.45*0.996*e2[j]<-944])#942.5
        #Removing samples that are outlying the straight line fit
        x = cat(x, x2[:,:,:, setdiff(1:end, ind)], dims=4)
        lp = cat(lp, lp2[setdiff(1:end, ind)], dims=1)
        E = cat(E, e2[setdiff(1:end, ind)], dims=1)
    end

    ptr = 1
    acc = 0
    mag = Array{Float64}(undef, size(E)...)
    mag[1] = mean(x[:,:,:,1])
    for j in 2:size(E)[1]
        arg = beta*(E[ptr]-E[j]) + (lp[ptr]-lp[j])
        if rand() < exp(arg)
            ptr = j
            acc += 1
        end
        mag[j] = mean(x[:,:,:,ptr])
    end
    @info "Acceptance Ratio" acc/(size(E)[1]-1.) size(E)[1]
    Magnetization = mean(abs.(mag))
    return Magnetization
end

mag = smart_mc(model, 32, 400, 0.45, n=10)
