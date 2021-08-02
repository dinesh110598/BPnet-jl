##
"""
    module BPnetModel
Module that builds and exports the BPnet object and useful, related funcitonalities
like sampling and log probability evaluators
"""
module BPnetModel
using Flux
using CUDA
using CUDA: tanh, Core, size

export BPnet, sample, log_prob, ZeroPad

"""
Pads zeros to arr at dim(=1 or 2), with the arg "before" determining whether
the padding is before or after. l indicates the length of padding
"""
function ZeroPad(arr::AbstractArray, dim::Integer, l::Integer, before=true)
    #Create array of zeros that will be padded to arr
    if dim==1
        z = similar(arr, l,0,size(arr)[3:4]...)
        fill!(z, 0f0)
    elseif dim==2
        z = similar(arr, 0,l,size(arr)[3:4]...)
        fill!(z, 0f0)
    end

    if before
        arr = cat(z, arr; dims=1:2)
    else
        arr = cat(arr, z; dims=1:2)
    end
    return arr
end

struct ConvBlock
    n::Int32
    p::Int32
    res::Bool #It seems safe to put these non-trainable layer parmeters in here
    last_layer::Bool
    l_conv::Tuple
    r_conv::Tuple
    m_conv::Tuple
end

function ConvBlock(p::Int, n::Int, mask::Char, last_layer)
    #mutable types like arrays are passed by reference
    #! is just a convention
    if mask=='A'
        res = 0
    elseif mask=='B'
        res = 1
    end
    l_conv = Conv((1, n-1+res), (mask=='A' ? 1 : p)=> 2*p, relu)
    l_conv2 = Conv((1, 1), 2*p => p, tanh)
    
    r_conv = Conv((1, n), (mask=='A' ? 1 : p)=> 2*p, relu)
    r_conv2 = Conv((1, 1), 2*p => p, tanh)
    
    m_conv = Conv((1,1), (mask=='A' ? 1 : p) => 2*p, tanh; bias=false)
    m_conv2 = Conv((1,1), p => p, sigmoid)
    res_conv = Conv((1,1), p => p, sigmoid; bias=false)
    return ConvBlock(n, p, res==1 ? true : false, last_layer, (l_conv, l_conv2), 
        (r_conv, r_conv2), (m_conv, m_conv2, res_conv))
end

Flux.@functor ConvBlock

"""Overall, calling this 'block' on the inputs x_l, x_m and x_r
acts them with a bunch of padding and convolution operations
that ultimately go into computing a probability for each site 
in x_m conditioned on the values in x_l and x_r. Many such
blocks are combined to make the BPnet defined later"""
function (block::ConvBlock)(x_l, x_m, x_r)

    if block.res
        x_m2 = copy(x_m)
    else
        x_l = x_l[:,1:end-1,:,:]#cropping x_l
        x_r = x_r[1:end-1, :, :, :]#cropping x_r
        x_r = ZeroPad(x_r, 1, 1, true)
    end
    x_l = ZeroPad(x_l, 2, block.n - 1, true) #Padding left stack
    x_r = ZeroPad(x_r, 2, block.n - 1, false) #Padding right stack
    
    x_l = block.l_conv[1](x_l) #Convolution on x_l, left stack
    x_r = block.r_conv[1](x_r) #Convolution on x_r, right stack
    x_m = block.m_conv[1](x_m) #Convolution on x_m
    x_m = x_m .+ x_l .+ x_r #Combining information in main stack
    
    x_m0 = tanh.(x_m[:, :, 1:end÷2, :]) #splitting x_m across channels dim
    x_m1 = σ.(x_m[:, :, (end÷2)+1:end, :])
    x_m = x_m0 .* x_m1 #Gating on x_m, main stack

    x_m = block.m_conv[2](x_m) #Further conv on x_m
    if block.res
        x_m2 = block.m_conv[3](x_m2)
        x_m = x_m .+ x_m2
    end
    if block.last_layer==false #Convolve left, right stacks if not last layer
        x_l = block.l_conv[2](x_l)
        x_r = block.r_conv[2](x_r)
    end
    return x_l, x_m, x_r
end

"""
    BPnet(kernel_size::Int, net_width::Int, net_depth::Int)

Initialize BPnet object by passing the following parameters

kernel_size: window size of convolution layers used

net_width: no of output channels in convolution layers

net_depth: no of ConvBlock layers used
"""
struct BPnet
    n::Int32
    p::Int32
    d::Int32
    layers::Array{ConvBlock}
    final_conv::Conv
end

function BPnet(kernel_size::Int, net_width::Int, net_depth::Int)
    layers = Array{ConvBlock}(undef, net_depth)
    for i in 1:net_depth
        layers[i] = ConvBlock(net_width, kernel_size, i==1 ? 'A' : 'B', 
                                    i==net_depth ? true : false)        
    end
    final_conv = Conv((1,1), net_width => 1, sigmoid)
    return BPnet(kernel_size, net_width, net_depth, layers, 
            final_conv)
end

Flux.@functor BPnet

function (model::BPnet)(x_l, x_m, x_r)
    for i in 1:model.d
        x_l, x_m, x_r = model.layers[i](x_l, x_m, x_r)
    end
    return model.final_conv(x_m)
end

"""
    (model::BPnet)(x_l, x_m, x_r)
Evaluates the conditional probability of +1 on each postion depending upon
the values at preceding positons in the raster scan order.

x: Input array based on which output probs are evaluated
"""
function (model::BPnet)(x)
    x_l = x
    x_r = copy(x_l)
    x_m = similar(x)
    fill!(x_m, 0f0)
    return model(x_l, x_m, x_r)
end

"""
    BernoulliGPU(prob::CuArray)

Returns a sample of Bernoulli distribution parameterized by, and the
same size as prob:

prob::CuArray - A CUDA array describing the probability parameters
of the distribution
"""
function BernoulliGPU(prob::CuArray)
    rnum = CUDA.rand(size(prob)...)
    2.0f0 .*(rnum .< prob) .- 1.0f0
end

function sample(model::BPnet, L::Integer, batch_size::Integer)
    sample = CUDA.zeros(L, L, 1, batch_size)
    r = model.d*(model.n - 1)
    for i in 1:L, j in 1:L
        #Slices upto 2 rows and 2*r+1 columns near i,j position
        x_l = sample[max(i-1,1):i, max(j-r, 1):min(j+r, L), :, :]
        x_r = copy(x_l)
        x_m = similar(x_l)
        fill!(x_m, 0f0)
        x_hat = model(x_l, x_m, x_r)
        i_h = min(i,2)
        j_h = min(j,r+1)
        #Bernoulli distribution sampling
        if (i, j) == (1, 1)
            probs = CUDA.fill(0.50f0, (1, batch_size))
        else
            probs = x_hat[i_h, j_h, :, :]
        end
        sample[i, j, :, :] = BernoulliGPU(probs)
    end
    
    #Enforce Z2 symmetry
    probs = CUDA.fill(0.5f0, (1,1,1,batch_size))
    sample = sample .* BernoulliGPU(probs)
end

function _log_prob(sample, x_hat)
    mask = (sample .+ 1.0f0) ./ 2.0f0
    log_prob = (
        CUDA.log.(x_hat .+ 1f-5).*mask +
        CUDA.log.(1.0f0 .- x_hat .+ 1f-5).*(1.0f0 .- mask)
    )
    log_prob = CUDA.sum(log_prob, dims=1:3)
    return reshape(log_prob, size(sample)[4])
end

function log_prob(model, sample)
    
    x_hat = model(sample)
    log_prob = _log_prob(sample, x_hat)
    #Enforcing Z2 symmetry
    sample_inv = -sample
    x_hat_inv = model(sample_inv)
    log_prob_inv = _log_prob(sample_inv, x_hat_inv)
    #Stacking log_prob and log_prob_inv along a horizontal dim
    z = cat(log_prob, log_prob_inv, dims=2)
    log_prob = reshape(logsumexp(z, dims=2), size(sample)[4])
    return log_prob
end

end
##
