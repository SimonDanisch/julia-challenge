import Base: getindex, iterate, axes, eachindex, tail
struct LazyBroadcast{F, Args}
    f::F
    args::Args
end
macro ibounds(f) #shortcut for the inlining & inbounds propagation
    esc(:(@inline Base.@propagate_inbounds $(f)))
end
getindex_arg(args::Tuple{}, I) = () # recursion ancor
@ibounds function getindex_arg(args::NTuple{N, Any}, I) where N
    return (br_getindex(args[1], I), getindex_arg(tail(args), I)...)
end
@ibounds function br_getindex(A::AbstractArray, I)
    idx = ntuple(i-> ifelse(size(A, i) === 1, 1, I[i]), Val(ndims(A)))
    return A[CartesianIndex(idx)]
end
br_getindex(scalar, I) = scalar # Scalars no need to index them
@ibounds function br_getindex(x::LazyBroadcast, I)
    return x.f(getindex_arg(x.args, I)...)
end
@ibounds getindex(x::LazyBroadcast, I) = br_getindex(x, Tuple(I))
function materialize!(out::AbstractArray, x::LazyBroadcast)
    @simd for i in CartesianIndices(axes(out))
        @inbounds out[i] = x[i]
    end
    return out
end
br_construct(x) = x
function br_construct(x::Expr)
    x.args .= br_construct.(x.args) # apply recursively
    if Meta.isexpr(x, :call) # replace calls to construct LazyBroadcasts
        x = :(LazyBroadcast($(x.args[1]), ($(x.args[2:end]...),)))
    end
    x
end
macro broadcast(call_expr)
    esc(br_construct(call_expr))
end
br = @broadcast a + b - sin(c)


using BenchmarkTools

reference(out, a, b, c) = (out .= a .+ b .- sin.(c))

a = rand(1000, 1000);
b = rand(1000);
c = 1.0
out = similar(a);
br = @broadcast a + b - sin(c)

@btime materialize!($out, $br)
@btime reference($out, $a, $b, $c)


# Library A - I chose GeometryTypes.
# Any library with NVectors specializing to the length will do!
using GeometryTypes
const Point3 = Point{3, Float32}

# function needs to come from different library
module LibraryB
  # no odd stuff, no functors, no special lambda expression!
  # this function needs to be a normal language function
  # as can be found in the wild
	super_custom_func(a, b) = sqrt(sum(a .* b))
end
# emulate that the function comes from a different library
using .LibraryB: super_custom_func

using BenchmarkTools

a = rand(Point3, 10^6)
b = rand(Point3, 10^6)
out = fill(0f0, 10^6)

@btime $out .= super_custom_func.($a, $b)
br = @broadcast super_custom_func(a, b)
@btime materialize!($out, $br)


# Simplified implementation to take the axes of the array with the largest
# dimensionality (axes -> the range an array iterates over)
biggest(a, b, c, rest...) = biggest(biggest(a, b), biggest(c, rest...))
biggest(a::NTuple{N1, Any}, b::NTuple{N2, Any}) where {N1, N2} =
	ifelse(N1 > N2, a, b)
biggest(a) = a
flatten_args(t::LazyBroadcast, rest...) =
	(flatten_args(t.args...)..., flatten_args(rest...)...)
flatten_args(t::Any, rest...) = (t, flatten_args(rest...)...)
flatten_args() = ()
axes(br::LazyBroadcast) = biggest(map(axes, flatten_args(br))...)
eachindex(br::LazyBroadcast) = CartesianIndices(axes(br))

iterate(br::LazyBroadcast) = iterate(br, (eachindex(br),))
@ibounds function iterate(bc::LazyBroadcast, s)
    y = iterate(s...)
    y === nothing && return nothing
    i, newstate = y
    return (bc[i], (s[1], newstate))
end

@btime sum($br)
@btime sum($a .+ $b .- sin.($c))


function threaded_materialize!(out::AbstractArray, x::LazyBroadcast)
    Threads.@threads for i in CartesianIndices(axes(out))
        @inbounds out[i] = x[i]
    end
    return out
end

a = @benchmark threaded_materialize!($out, $br)
b = @benchmark materialize!($out, $br)
hyper_threads = Sys.CPU_THREADS
# intels hyper threads don't really count in such intense calculations
cores = hyper_threads / 2
# scaling
println("best scaling:", minimum(b).time / minimum(a).time / cores)
println("average scaling:", mean(b).time / mean(a).time / cores)
