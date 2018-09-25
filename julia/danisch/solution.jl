import Base: getindex, iterate, axes, eachindex, tail, @propagate_inbounds
struct LazyBroadcast{F, Args}
    f::F
    args::Args
end
@propagate_inbounds function br_getindex(A::AbstractArray, I)
    idx = ntuple(i-> ifelse(size(A, i) === 1, 1, I[i]), Val(ndims(A)))
    return A[CartesianIndex(idx)]
end
br_getindex(scalar, I) = scalar # Scalars no need to index them
@propagate_inbounds function br_getindex(x::LazyBroadcast, I)
    # this could be a map, but the current map in 1.0 has a perf problem
    return x.f(getindex_arg(x.args, I)...)
end
getindex_arg(args::Tuple{}, I) = () # recursion ancor
@propagate_inbounds function getindex_arg(args::NTuple{N, Any}, I) where N
    return (br_getindex(args[1], I), getindex_arg(tail(args), I)...)
end
@propagate_inbounds getindex(x::LazyBroadcast, I) = br_getindex(x, Tuple(I))
function materialize!(out::AbstractArray, x::LazyBroadcast)
    # an n-dimensional simd accelerated loop
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
# macro to enable the syntax @broadcast a + b - sin(c) to construct our type
macro broadcast(call_expr)
    esc(br_construct(call_expr))
end
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
# the indexing axes of our array
axes(br::LazyBroadcast) = biggest(map(axes, flatten_args(br))...)
# lazy view that can be used to index over all elements in br
eachindex(br::LazyBroadcast) = CartesianIndices(axes(br))
iterate(br::LazyBroadcast) = iterate(br, (eachindex(br),))
@propagate_inbounds function iterate(bc::LazyBroadcast, s)
    y = iterate(s...)
    y === nothing && return nothing
    i, newstate = y
    return (bc[i], (s[1], newstate))
end
