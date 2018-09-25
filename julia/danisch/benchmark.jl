# Benchmarks

using BenchmarkTools

reference(out, a, b, c) = (out .= a .+ b .- sin.(c))

a = rand(1000, 1000);
b = rand(1000);
c = 1.0
out = similar(a);
br = @broadcast a + b - sin(c)

@btime materialize!($out, $br)
@btime reference($out, $a, $b, $c)

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

@btime sum($br)
@btime sum($a .+ $b .- sin.($c))
