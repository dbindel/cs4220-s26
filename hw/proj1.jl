### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ 8498e1f6-6c9b-486f-9b68-7295b119e080
using LinearAlgebra

# ╔═╡ f9e57c29-bc78-4e8d-b4f7-3167ea54bc4b
using Plots

# ╔═╡ ecf1d95f-9994-4551-b287-44ad9458b5d3
using SparseArrays

# ╔═╡ eba1273a-5100-4a13-b7cd-72e619732231
using SuiteSparse

# ╔═╡ c179b6ac-06aa-46c0-8be1-8044c6bc807f
using SpecialFunctions

# ╔═╡ 952498f6-0aaa-11f1-b51c-39d3a1e83ab9
md"""
# Proj 1: Approximation with kernels

A classic function approximation scheme used for interpolating data at scattered points involves the use of a *radial basis function* (RBF), typically denoted by $\phi$.  For a function $f : \mathbb{R}^d \rightarrow \mathbb{R}$, we approximate $f$ by

$$s(x) = \sum_{i=1}^n \phi(\|x-x_i\|_2) c_i$$

where the points $\{x_i\}_{i=1}^n$ are known as *centers* -- we will assume for the purpose of this assignment that these are all distinct.  We sometimes write this more concisely as

$$s(x) = \Phi_{xX} c$$

where $\Phi_{xX}$ is a row vector with entries $(\Phi_{xX})_j = \phi(\|x-x_i\|_2)$.

The coefficient vector $c$ may be chosen by interpolating at the centers; that is, we write $s(x_i) = f(x_i),$ or more compactly

$$\Phi_{XX} c = f_X$$

where $\Phi_{XX} \in \mathbb{R}^{n \times n}$ is the matrix with entries $(\Phi_{XX})_{ij} = \phi(\|x_i-x_j\|)$ and $f_X \in \mathbb{R}^n$ is the vector with entries $(f_X)_i = f(x_i)$.  When $\phi$ is a *positive definite* RBF, the matrix $\Phi_{XX}$ is guaranteed to be positive definite.

There are many reasons to like RBF approximations.  There is a great deal of theory associated with them, both from a classic approximation theory perspective and from a statistics perspective (where $\phi$ is associated with the covariance of a *Gaussian process*).  But for this project, we also like RBF approximations because they naturally give rise to many different types of numerical linear algebra problems associated with solving linear systems!  We will explore some of these in the current project.
"""

# ╔═╡ 488686f6-3781-43ac-b5b9-2366934e69ec
md"""
## Logistics

You should ideally complete tasks 1 and 2 by Monday, Feb 23; and tasks 3 and 4 by Monday, Mar 2.

You are encouraged to work in pairs on this project. I particularly encourage you to try pair-thinking and pair-programming -- you learn less if you just try to partition the problems!  You should produce short report addressing the analysis tasks, and a few short codes that address the computational tasks. You may use any Julia functions you might want.  Please make your write-up self-contained: it should include your (short) codes as well as any requested discussion, plots, and theory.

You are allowed (even encouraged!) to read outside resources that talk about these types of computations -- including [my own notes](https://www.cs.cornell.edu/courses/cs6241/2025sp/lec/2025-03-11.html) from my ["Numerical Methods for Data Science" course](https://www.cs.cornell.edu/courses/cs6241/2025sp/).
I also have a set of [notes on kernel computations and Bayesian optimization](https://bindel-group.github.io/kernel-basics/) that may be useful.
It is possible that you'll find references that tell you outright how to solve a subproblem; if so, feel free to take advantage of them *with citation*!  You may well end up doing more work to find and understand the relevant resources than you would doing it yourself from scratch, but you will learn interesting things along the way.

Most of the code in this project will be short, but that does not make it easy. You should be able to convince both me and your partner that your code is right. A good way to do this is to test thoroughly. Check residuals, compare cheaper or more expensive ways of computing the same thing, and generally use the computer to make sure you don't commit silly errors in algebra or coding. You will also want to make sure that you satisfy the efficiency constraints stated in the tasks.
"""

# ╔═╡ fa395782-bee0-4c61-b649-8ccbca40cca1
md"""
## Code setup

We will use several built-in packages for this project.  We'll use the `LinearAlgebra` and `Plots` packages in all most all of our computational homeworks and projects in this course, but here we also use the `SparseArrays` and `SuiteSparse` packages for dealing with sparse linear algebra.  We also use the `SpecialFunctions` package for the definition of `erf`.
"""

# ╔═╡ 212a44cf-ec9e-4952-a065-60194be0106f
md"""
### Basis functions

There are many possible basis functions, but we limit our attention to two: the squared exponential function (sometimes known confusingly as "the" RBF function) and the compactly supported Wendland function.  The `scale_rbf` function turns a radial basis function with a default lengths scale of 1 into an RBF with a length scale.
"""

# ╔═╡ 58d88ea1-1f6d-4f56-998a-e94c709943ba
begin
	ϕ_se(r) = exp(-r^2)
	ϕ_w21(r) = max(1.0-r, 0.0)^4*(4.0*r+1.0)
	scale_rbf(ϕ, l=1.0) = r->ϕ(r/l)
end

# ╔═╡ 8a5692af-e265-429c-96ab-f78dd409a113
md"""
### Sampling the space

When we want to sample something in more than one spatial dimension and aren't just using a regular mesh, it is tempting to choose random samples.  But taking independent uniform draws is not an especially effective way of covering a space – random numbers tend to clump up. For this reason, [low discrepancy sequences](https://en.wikipedia.org/wiki/Low-discrepancy_sequence) are often a better basis for sampling than (pseudo)random draws. There are many such generators; we use a relatively simple one based on an additive recurrence with a multiplier based on the ["generalized golden ratio"](http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/).
"""

# ╔═╡ e310d6b8-ed68-49db-aa36-9d2040fed799
"""
    X = kronecker_quasirand(d, N, start=0)

Returns a `d`-by-`N` array `X` whose columns are drawn from the `d`-dimensional
Kronecker sequence starting at index `start`.
"""
function kronecker_quasirand(d, N, start=0)
    
    # Compute the recommended constants ("generalized golden ratio")
    ϕ = 1.0+1.0/d
    for k = 1:10
        gϕ = ϕ^(d+1)-ϕ-1
        dgϕ= (d+1)*ϕ^d-1
        ϕ -= gϕ/dgϕ
    end
    αs = [mod(1.0/ϕ^j, 1.0) for j=1:d]
    
    # Compute the quasi-random sequence
    Z = zeros(d, N)
    for j = 1:N
        for i=1:d
            Z[i,j] = mod(0.5 + (start+j)*αs[i], 1.0)
        end
    end
    
    Z
end

# ╔═╡ 3f464c20-78b5-46b5-87cc-14e405a694b6
md"""
### Finding neighbors

If we work with a radial basis function $\phi$ that is zero past some distance $r_{\mathrm{cutoff}}$, then $(\Phi_{XX})_{ij} \neq 0$ only if $\|x_i-x_j\| \leq r_{\mathrm{cutoff}}$.  We can, of course, find all such pairs in $O(n^2)$ time by checking all possible pairs.  But things can go faster with a simple data structure.  In 2D, think of putting every point $x_i$ into a square "bucket" with side length $r_{\mathrm{cutoff}}$.  Then all points within $r_{\mathrm{cutoff}}$ of $x_i$ must be in one of nine buckets: the one containing $x_i$, or any of its neighbors.  If we give the buckets identifiers and sort them in "column major" order (and sort their contents accordingly), this can significantly reduce the number of potential neighbors to any point that we might have to check.
"""

# ╔═╡ 47f69b9f-2118-42ba-8d36-b84f0750c31f
"""
    xy_new, p, neighbors, I, J, r = find_neighbors2d(xy, rcutoff)

Given a 2-by-n matrix `xy` of coordinates and a cutoff threshold,
quickly find all pairs of points within the cutoff from each other.
Returns
 - `xy_new` -- a reordered version of the input matrix in "bucket order"
 - `p` -- the reordering to bucket order (`xy_new = xy[:,p]`)
 - `neighbors(q) -> Iq, rq` -- a function that takes a query point `q` and finds
   indices `Iq` of points within the cutoff of `q` and `rq` of the distances from `q`
   to those points.  The index is with respect to the bucket order.
 - `I`, `J`, `r` -- three parallel arrays such that `I[k]`, `J[k]`, `r[k]`
   represents the indices of two points within the cutoff of each other and 
   the distance between them.
"""
function find_neighbors2d(xy, rcutoff)

	# Logic for computing integer bucket indices
	n = size(xy)[2]
	xmin = minimum(xy[1,:])
	xmax = maximum(xy[1,:])
	ymin = minimum(xy[2,:])
	ymax = maximum(xy[2,:])
	idx1d(x) = floor(Int, x/rcutoff) + 1
	nx = idx1d(xmax-xmin)
	idx(x,y) = idx1d(x-xmin) + (idx1d(y-ymin)-1)*nx

	# Assign points to buckets, sort by bucket index
	buckets = idx.(xy[1,:], xy[2,:])
	p = sortperm(buckets)
	buckets = buckets[p]
	xy = xy[:,p]

	# Set up three parallel arrays for tracking neighbors closer than the cutoff
	I = zeros(Int, n)
	J = zeros(Int, n)
	r = zeros(Float64, n)
	I[:] = 1:n
	J[:] = 1:n
	function process_edge(i, j)
		rij = norm(xy[:,i]-xy[:,j])
		if rij <= rcutoff
			push!(I, i); push!(J, j); push!(r, rij)
			push!(I, j); push!(J, i); push!(r, rij)
		end
	end

	# Use the bucket structure to check all pairwise interactions quickly
	# (process each edge once, self-loops handled separately).  Note it's
	# fine if we "wrap around" the edge of the bucket array -- it just means
	# checking some extra budgets.
	for i = 1:n
		j0 = searchsortedfirst(buckets, buckets[i]-nx-1)
		j1 = searchsortedlast(buckets, buckets[i]-nx+1)
		for j = j0:j1
			process_edge(i, j)
		end
		j2 = searchsortedfirst(buckets, buckets[i]-1)
		for j = j2:i-1
			process_edge(i, j)
		end
	end

	# Find neighbors of a query point
	function neighbors(qxy)
		b = idx(qxy[1], qxy[2])
		nbrs = Int[]
		rs = Float64[]
		for col = -1:1
			j0 = searchsortedfirst(buckets, b+col*nx-1)
			j1 = searchsortedlast(buckets, b+col*nx+1)
			for j = j0:j1
				rqj = norm(qxy-xy[:,j])
				if rqj <= rcutoff
					push!(nbrs, j)
					push!(rs, rqj)
				end
			end
		end
		nbrs, rs
	end

	# Return permuted coordinates, etc
	xy, p, neighbors, I, J, r
end

# ╔═╡ 1e466e15-50cc-43c2-be8c-739967f5d424
md"""
### Hello world examples

To illustrate how to code a kernel approximation scheme, consider the problem of interpolating $\cos(x)$ using 8 samples on the interval $[0, 2\pi]$.  What is the error in the resulting RBF approximation over the interval, assuming we use a squared exponential kernel with length scale 1?  Written to take advantage of Julia's language feature, the code to answer this question is scarcely longer than the text to describe the question!
"""

# ╔═╡ f912e938-1270-4911-9eef-94c857ab0193
function hello_test1d(npts=8)
	# Set up approximation points
	x = range(0, 2π, length=npts)

	# Set up and solve kernel system
	Φxx = [ϕ_se(norm(xi-xj)) for xi in x, xj in x]
	c = Φxx\cos.(x)
	s(z) = sum(ϕ_se(norm(xi-z))*ci for (xi,ci) in zip(x, c))

	# Plot error on finer mesh
	xx = range(0, 2π, length=100)
	plot(xx, cos.(xx)-s.(xx), linewidth=4, label="Error")
end

# ╔═╡ c4c36151-4781-480b-9113-43e43fbf20fe
hello_test1d(8)

# ╔═╡ b4a97d3f-1d28-4fb3-9a18-82c2df721bfb
md"""
We also give an example that does something similar in 2D, though here we only bother to look at the center point error, and we are using the Wendland kernel with a limited cutoff radius that depends on the point density.
"""

# ╔═╡ 40343cef-fd41-4215-884e-bd0cc9eac711
function hello_test2d(npoints=10000)
	xy = kronecker_quasirand(2, npoints)

	# Set up kernel and sparse kernel matrix
	l = 5.0/sqrt(npoints)
	ϕ = scale_rbf(ϕ_w21, l)
	@time xy, p, neighbors, I, J, r = find_neighbors2d(xy, l)
	@time ΦXX = sparse(I, J, ϕ.(r))
	
	# Solve test problem
	ftest(x) = cos(x[1])*exp(x[2])
	@time c = ΦXX\[ftest(xi) for xi in eachcol(xy)]
	s(z) = sum(ϕ(norm(z-xi))*ci for (xi,ci) in zip(eachcol(xy),c))

	# Look at error at center point
	rerr = (ftest([0.5; 0.5])-s([0.5; 0.5]))/ftest([0.5;0.5])
	md"Relative error at center is: $rerr"
end

# ╔═╡ 315f6658-bd33-404c-8d01-55888e41bcab
hello_test2d()

# ╔═╡ 0e496bad-a1d1-4c49-8c00-1b74691453b9
md"""
## Tasks

You will have four tasks, each involving a couple questions.  We also indicate the approximate number of points per subtask (out of 10).
"""

# ╔═╡ 8cb8a9f3-ef47-4a77-b1c7-2cfacadd15c8
md"""
### Task 1: Fast mean temperatures (1 point)

So far, we have discussed approximating one function at many points.  Sometimes, we would like to quickly approximate something about *many* functions.  For example, in 1D, suppose we have streaming temperature measurements $\theta_j(t)$ taken at $n$ fixed points $x_j \in [0, 1]$, from which we could estimate an instantaneous temperature field

$$\hat{\theta}(x, t) = \sum_{j=1}^n \phi(\|x-x_j\|) c_j$$

by the interpolation condition $\hat{\theta}(x_i, t) = \theta_i(t)$.  Assuming we have a factorization for $\Phi_{XX}$, we can compute the estimated mean temperature

$$\tilde{\theta}(t) = \int_0^1 \hat{\theta}(x, t) \, dx = \sum_{j=1}^n \left( \int_0^1 \phi(\|x-x_j\|) \, dx \right) c_j$$

for any given $t$ by solving for the $c$ vector ($O(n^2)$ time) and taking the appropriate linear combination of integrals ($O(n)$ time).  Here it is worth noting that

$$\int_0^1 \phi(\|x-x_j\|) \, dx = \int_0^{x_j} \phi(s) \, ds + \int_{0}^{1-x_j} \phi(s) \, ds$$

and for $\phi(r) = \exp(-r^2/l^2)$, we have

$$\int_0^x \phi(s) \, ds = \frac{l \sqrt{\pi}}{2} \operatorname{erf}(x/l),$$

where the error function $\operatorname{erf}(x)$ is implemented as `erf(x)` in the Julia `SpecialFunction` library.  We implement this scheme in the following code, which runs in $O(n^3) + O(n^2 m)$ time with $n$ sensors and $m$ measurements.
"""

# ╔═╡ 237b7b00-98b3-4339-8ebc-a8861ff4b582
"""
    Θmeans = mean_temperatures(x, Θ)

Given a set of sensor locations x and a time series of temperature measurements
Θ[i,j] = measurement of sensor i at time j, return the corresponding time series
of mean temperature estimates Θmeans.
"""
function mean_temperatures(x, Θ)

	# Set up the RBF and RBF matrix for interpolation
	l = 1.0/length(x)
	ϕ = scale_rbf(ϕ_se, l)
	ΦXX = [ϕ(norm(xi-xj)) for xi in x, xj in x]

	# Computation for weights
	wt(x) = l*sqrt(π)/2 * (erf(x/l) + erf((1.0-x)/l))
	w = [wt(xi) for xi in x]

	# Compute the mean temperature for each column
	cs = ΦXX\Θ
	cs'*w
end

# ╔═╡ bdb29c97-e992-441d-b8c0-72d037c01e96
md"""
An example run with about 2000 virtual sensors and 5000 samples takes a bit under 1.2 seconds to process this way on my laptop.
"""

# ╔═╡ a6633370-4ddc-41af-85f8-b92f88f1e06d
let
	nsensor = 2001
	ntimes = 5000
	
	x_sensor = zeros(nsensor)
	x_sensor[:] = range(0, 1, length=nsensor)
	Θ = [cos(x * 2π * j/ntimes) + 10.0*j/ntimes for x in x_sensor, j in 1:ntimes]

	θmean = mean_temperatures(x_sensor, Θ)
	plot(θmean, xlabel="time", ylabel="mean temp", legend=false)
end

# ╔═╡ 3872e2e2-792d-4a98-a025-205493aa381c
md"""
**TODO** (1 point): Rewrite this code to take $O(n^3) + O(mn)$ time by reassociating the expression $\tilde{\theta}(t) = w^T \Phi_{XX}^{-1} \theta_X(t)$.  Compare the timing of the example given above to the timing of your routine; do the numbers seem reasonable?  At the same time, do a comparison of the results to make sure that you do not get something different!  You may want to use the testing and timing harness below.
"""

# ╔═╡ b09b0a80-ce1e-466c-96f2-7855a9c0abb1
function mean_temperatures2(x, Θ)

	# Set up the RBF and RBF matrix for interpolation
	l = 1.0/length(x)
	ϕ = scale_rbf(ϕ_se, l)
	ΦXX = [ϕ(norm(xi-xj)) for xi in x, xj in x]

	# Computation for weights
	wt(x) = l*sqrt(π)/2 * (erf(x/l) + erf((1.0-x)/l))
	w = [wt(xi) for xi in x]

	# TODO: Replace this with your fast implementation (1 point)
	cs = ΦXX\Θ
	cs'*w
end

# ╔═╡ 4dd4bb10-9d10-402b-a781-c4b6c3e01187
function test_sensor_demo(nsensor, ntimes)
	x_sensor = range(0, 1, length=nsensor)
	Θ = [cos(x * 2π * j/ntimes) + 10.0*j/ntimes for x in x_sensor, j in 1:ntimes]

	@time θmean1 = mean_temperatures(x_sensor, Θ)
	@time θmean2 = mean_temperatures2(x_sensor, Θ)
	norm(θmean1-θmean2, Inf)
end

# ╔═╡ e546ec33-ee20-4e6e-a492-ef74d9f9338a
md"""
### Task 2: Missing data (2 points)

Now suppose that we are again dealing with streaming sensor data, but every so often there are entries missing.  If the $k$th measurement is missing, the interpolation conditions for the remaining points can be written in terms of a smaller system of equations where we remove row and column $k$ from the original problem; *or* as

$$\begin{align*}
  f(x_i) &= \sum_{j=1}^n \phi(\|x_i-x_j\|) \hat{c}_j + r_i, & r_i = 0 \mbox{ for } i \neq k  \\
  \hat{c}_k &= 0.
\end{align*}$$

Equivalently, we have

$$\begin{bmatrix}
  \Phi_{XX} & e_k \\
  e_k^T & 0
\end{bmatrix} 
\begin{bmatrix} \hat{c} \\ r_k \end{bmatrix}
=
\begin{bmatrix} \tilde{f}_X \\ 0 \end{bmatrix}$$

where $\tilde{f}_X$ agrees with $f_X$ except in the $k$th element.  If we set $(\tilde{f}_X)_k = 0$, then $-r_k$ is the value at $x_k$ of the interpolant through the remaining data -- potentially a pretty good guess for the true value.

**TODO** (2 points): Using block elimination on the system above, complete the following routine to fill in a single missing value in an indicated location.  Your code should take $O(n^2)$ time (it would take $O(n^3)$ to refactor from scratch).  Also provide a short test to demonstrate correctness of your code.
"""

# ╔═╡ df724218-18d0-4192-9b4e-1e440ae846f9
"""
    fX[k] = fill_missing(!F, fX, k)

Given a Cholesky factorization object `F` for the RBF matrix `ΦXX` and a right-hand-side vector `fX` where `fX[k]` is invalid, impute `fX[k]` from the other data points.
"""
function fill_missing!(F, fX, k)
	# TODO
end

# ╔═╡ 1bbaf624-9896-43c5-b316-008598b935b4
md"""
### Task 3: Pivoted Cholesky down-selection (3 points)

Suppose we are given a radial basis function $\phi$ and a compact domain $\Omega$.  How do we choose points in the domain that are "best" for approximation?  One approach is to use *Fekete points*, which maximize $\det \Phi_{XX}$.  However, this leads to a difficult optimization problem, and we sometimes are not in a position to choose all points simultaneously, but instead want to incrmentally *add* points until we have an approximation with which we are satisfied.  That is
suppose we have chosen a set of points $X$ as the basis of RBF approximation, and want to choose a new point $z$ from a set of candidates.  One way to do this is to maximize the determinant of the new RBF matrix:

$$\det \begin{bmatrix} \Phi_{XX} & \Phi_{Xz} \\ \phi_{zX} & \phi(0) \end{bmatrix}$$

This can be seen as a greedy approximation of the Fekete approach.

**TODO** (1 point): Argue that maximizing the determinant with respect to placement of $z$ is equivalent to maximizing the Schur complement $\phi(0) - \Phi_{zX} \Phi_{XX}^{-1} \Phi_{Xz}$
"""

# ╔═╡ 54cfa7e3-9e2b-4455-820d-eb3afe9229e1
md"""
The *pivoted Cholesky* algorithm can be used to greedily choose $k$ points out of $N$ candidates.  The standard pivoted Cholesky factorization requires $O(N^3)$ time and $O(N^2)$ space; the unblocked right-looking version of the algorithm looks like this (a more sophisticated version is can be called with `cholesky(A, RowMaximum())` in Julia).
"""

# ╔═╡ da903750-48b5-43d2-8070-a9fb9d70707c
"""
    p, R = basic_pivchol!(A)

Overwrite the upper triangle of input matrix `A` (assumed symmetric)
with the pivoted Cholesky factorization.  Return `p`, `R` s.t. `A[p,p] = R'*R`
"""
function basic_pivchol!(A)
	m, n = size(A)
	if m != n  throw(ArgumentError("Input matrix A should be square"))  end

	# Set up pivot array
	p = zeros(Int, n)
	p[:] .= 1:n

	for j = 1:n

		# Find pivot element
		Apiv, jpiv = findmax(A[l,l] for l=j:n)
		jpiv += j-1
		if Apiv <= 0 throw(ArgumentError("Matrix was not positive definite"))  end

		# Perform column and row swap
		p[j], p[jpiv] = p[jpiv], p[j]
		for i=1:n  A[i,j], A[i,jpiv] = A[i,jpiv], A[i,j]  end
		for i=1:n  A[j,i], A[jpiv,i] = A[jpiv,i], A[j,i]  end

		# Compute a row of R
		A[j,j] = sqrt(Apiv)
		A[j,j+1:n] /= A[j,j]

		# Schur complement update
		for l=j+1:n
			ajl = A[j,l]
			for i=j+1:n
				A[i,l] -= A[j,i]*A[j,l]
			end
		end
	end

	p, UpperTriangular(A)
end

# ╔═╡ f16d89e9-d659-4ee7-b5b6-dbfd58a19e7a
md"""
It is always useful to have a sanity check of such computations.
"""

# ╔═╡ e0bc890b-b30b-4820-bb3e-810739cb6e00
let
	A = randn(10,10)
	A = A'*A
	p, R = basic_pivchol!(copy(A))
	norm(A[p,p]-R'*R)/norm(A)
end

# ╔═╡ d5e02d2e-2ca2-41c6-b888-55efcebc0e4c
md"""
We can naively use `basic_pivchol!` to compute leading $k$ rows of the factorization to get the low-rank approximation $P \Phi_{XX} P^T \approx R_{12}^T R_{12}$ by computing the full pivoted Cholesky and throwing most of it away:
"""

# ╔═╡ 706f9016-275a-4c03-8877-cdfca8353292
"""
    p, R12 = naive_greedy_select(ϕ, X, k)

Compute the leading `k`-by-`N` part of the pivoted Cholesky factorization, i.e.
`ΦXX[p[1:k],p] = R12[:,1:k]'*R12`
"""
function naive_greedy_select(ϕ, X, k)
	p, R = basic_pivchol!([ϕ(norm(xi-xj)) for xi in eachcol(X), xj in eachcol(X)])
	p, R[1:k,:]
end

# ╔═╡ 64d33191-69e1-447c-b91a-2870d6bb26cb
md"""
A better approach, and your main goal for this task, is to write a modified version of pivoted Cholesky that computes only the leading $k$ rows and takes $O(kN)$ space and $O(k^2 N)$ time.  To do this, you will need to mostly defer Schur complement updates (as in a left-looking algorithm), except that you need to keep the diagonal of the kernel matrix in order to determine the pivot order.

**TODO**: Write an efficient version of `greedy_select` and test against the naive algorithm (2 points).
"""

# ╔═╡ 90df6819-e288-443f-99a1-796e96de85b7
"""
    p, R12 = greedy_select(ϕ, X, k)

Compute the leading `k`-by-`N` part of the pivoted Cholesky factorization, i.e.
`ΦXX[p[1:k],p] = R12[:,1:k]'*R12`.  This algorithm should take O(k^2 N) time
and O(kN) space.
"""
function greedy_select(ϕ, X, k)
	naive_greedy_select(ϕ, X, k)
end

# ╔═╡ b58646f4-35ba-40bc-aad6-9a0f054fc7cb
md"""
### Task 4: Sparse approximation (4 points)

So far, except in our "hello, world" exercise, we have used dense matrix representations for everything.  As you have seen, this is often basically fine for up to a couple thousand data points, provided that we are careful to re-use factorizations and organize our computations for efficiency.  When we have much more data, though, the $O(n^3)$ cost of an initial factorization gets to be excessive.

There are several ways to use *data sparsity* of the RBF matrix for faster (approximate) solves.  For this project, though, we will focus on a simple one: using ordinary sparsity of $\Phi_{XX}$ for compactly supported kernels like the Wendland, or approximating that sparsity for rapidly-decaying kernels like the squared exponential.  In the latter case, though, we will need to have some care, as we will see.
"""

# ╔═╡ 0f387faf-3ac3-48de-ab1f-68f6e943d11d
md"""
Let's now put together a version of our "hello world" 2D program with the squared exponential kernel with a truncation.
"""

# ╔═╡ 2a029c1d-ec2f-4731-b35d-59207a75ebe6
function hello_test2d_se(npoints=10000; σ=4.0, l=0.02)
	xy = kronecker_quasirand(2, npoints)

	# Set up kernel and sparse kernel matrix
	l = l > 0 ? l : 1.0/sqrt(npoints)
	ϕ = scale_rbf(ϕ_se, l)
	xy, p, neighbors, I, J, r = find_neighbors2d(xy, σ*l)
	ΦXX = sparse(I, J, ϕ.(r))
	
	# Solve test problem
	ftest(x) = cos(x[1])*exp(x[2])
	F = cholesky(ΦXX)
	c = F\[ftest(xi) for xi in eachcol(xy)]
	s(z) = sum(ϕ(norm(z-xi))*ci for (xi,ci) in zip(eachcol(xy),c))

	# Look at density of ΦXX and error at center point
	rerr = (ftest([0.5; 0.5])-s([0.5; 0.5]))/ftest([0.5;0.5])
	nnz(ΦXX)/npoints^2, rerr
end

# ╔═╡ 7177aa90-3646-45aa-8ba9-e3158b9352b7
md"""
**TODO** (1 point): Even a little playing with the code illustrates that it is delicate.  For example, what happens if $l = 0.05$?  If $\sigma = 3$?
"""

# ╔═╡ 44c295c4-4f6f-4ff8-9115-76264b2fdf9b
md"""
**TODO** (1 point): We would like to better understand the error in this approximation.  Let $\hat{s}(x)$ be the sparsified approximator that we have computed above, i.e.

$$\hat{s}(x) = \sum_{\|x-x_i\| \leq r_{\mathrm{cutoff}}} \phi(\|x-x_i\|) c_{0,i}$$

Let us start by showing that

$$|\hat{s}(x)-s(x)| \leq \|c_0\|_1 \exp(-\sigma^2) + \|c-c_0\|_1.$$

*Hint/sketch*: Write $\hat{\Phi}_{xX}$ as the vector of evaluations with cutoff, so that $\hat{s}(x) = \hat{\Phi}_{xX} c_0$ and $s(x) = \Phi_{xX} c$.  Add and subtract $\Phi_{xX} c_0$; apply the triangle inequality; use the fact that $|u \cdot v| \leq \|u\|_1 \|v\|_\infty$ for any $u$ and $v$; and derive simple bounds on $\|\Phi_{xX}\|_\infty$ and $\|\hat{\Phi}_{xX}-\Phi_{xX}\|_\infty$.
"""

# ╔═╡ 302e368d-d282-4cbb-946a-8e4f13048bf2
md"""
The bound is pessimistic, but it gives a sense of what can go wrong: if there is a large error in our coefficients, or if the coefficient norm times the magnitude of the neglected RBF evaluations is big, then we may have large differences between $s$ and $\hat{s}$.

The error in the coefficient vector $\|c-c_0\|_1$ with our initial parameters can be quite large, enough to make our error bounds terrible (even if the error might not be as bad).  But even if the factorization of $\hat{\Phi}_{XX}$ is not good enough to get a very accurate $c$ alone, it is good enough to make progress.  We therefore try an *iterative refinement loop*, combining our approximate solver based on $\hat{\Phi}_XX$ and a matrix vector product with a (still sparse) $\tilde{\Phi}_XX$ that provides a much more approximation to $\Phi_{xX}$.  Your goal: carry out the iterative refinement and give evidence of its convergence.
"""

# ╔═╡ d159a679-66cb-41d3-a864-5380cfb870bb
md"""
**TODO** (1.5 points): Fill in the iterative refinement loop and convince yourself (and us) that it is effective.  Comparing direct sparse computation with a large radius $6l$ to iterative refinement using a smaller radius $4l$ in the approximate factorization, what is the speed advantage?  What do you think the rationale is for using $6l$ as our "true" problem?

For the last 0.5 points: try something else!  Show that you can increase $l$ or decrease $\sigma$ (e.g. to $0.02$ or $3.0$) if $\eta$ is larger; what is the effect on error?  Or see how bad the error actually seems to be relative to the bounds.  Or do something else... anything to explore a bit.
"""

# ╔═╡ f353a3f3-fa2d-4280-a746-a6e2f2e208c3
function hello_test2d_se_itref(npoints=20000; σ=4.0, σ1=6.0, l=0.01, η=0.0, 
							   niter=12, verbose=false)
	xy = kronecker_quasirand(2, npoints)

	# Set up kernel and sparse kernel matrix
	ϕ = scale_rbf(ϕ_se, l)
	xy, p, neighbors, I, J, r = find_neighbors2d(xy, σ1*l)
	Iσ = (r .< σ*l)
	ϕr = ϕ.(r)
	ηd = zeros(size(r))
	ηd[I .== J] .+= η
	ΦXX0 = sparse(I[Iσ], J[Iσ], ϕr[Iσ] + ηd[Iσ])  # Coarser approximation
	ΦXX1 = sparse(I, J, ϕr)

	# Set up test problem
	ftest(x) = cos(x[1])*exp(x[2])
	fX = [ftest(xi) for xi in eachcol(xy)]

	# Initial solve
	F = cholesky(ΦXX0)
	c = F\fX

	# TODO: Fill in iterative refinement

	# Set up evaluation routine (don't worry about near neighbors)
	s(z) = sum(ϕ(norm(z-xi))*ci for (xi,ci) in zip(eachcol(xy),c))

	# Look at density of ΦXX and error at center point
	rerr = (ftest([0.5; 0.5])-s([0.5; 0.5]))/ftest([0.5;0.5])
	nnz(ΦXX0)/npoints^2, rerr
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
SuiteSparse = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[compat]
Plots = "~1.41.5"
SpecialFunctions = "~2.7.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.5"
manifest_format = "2.0"
project_hash = "640a8a6507be351c1875b28f56b85212752957b1"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "fde3bf89aead2e723284a8ff9cdf5b551ed700e8"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.5+0"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b0fd3f56fa442f81e0a47815c92245acfaaa4e34"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.31.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "d9d26935a0bcffc87d2613ce14c527c99fc543fd"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.5.0"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "e357641bb3e0638d353c4b29ea0e40ea644066a6"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.3"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "473e9afc9cf30814eb67ffa5f2db7df82c3ad9fd"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.16.2+0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a4be429317c42cfae6a7fc03c31bad1970c310d"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+1"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "d36f682e590a83d63d1c7dbd287573764682d12a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.11"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "27af30de8b5445644e8ffe3bcb0d72049c089cf1"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.7.3+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "95ecf07c2eea562b5adbd0696af6db62c0f52560"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.5"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "01ba9d15e9eae375dc1eb9589df76b3572acd3f2"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "8.0.1+0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "f85dac9a96a01087df6e3a749840015a0ca3817d"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.17.1+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "2c5512e11c791d1baed2049c5652441b28fc6a31"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "b7bfd56fa66616138dfe5237da4dc13bbd83c67f"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.1+0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "ee0585b62671ce88e48d3409733230b401c9775c"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.22"

    [deps.GR.extensions]
    IJuliaExt = "IJulia"

    [deps.GR.weakdeps]
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "7dd7173f7129a1b6f84e0f03e0890cd1189b0659"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.22+0"

[[deps.GettextRuntime_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll"]
git-tree-sha1 = "45288942190db7c5f760f59c04495064eedf9340"
uuid = "b0724c58-0f36-5564-988d-3bb0596ebc4a"
version = "0.22.4+0"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Zlib_jll"]
git-tree-sha1 = "38044a04637976140074d0b0621c1edf0eb531fd"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.1+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "24f6def62397474a297bfcec22384101609142ed"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.86.3+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a6dbda1fd736d60cc477d99f2e7a042acfa46e8"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.15+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "5e6fe50ae7f23d171f44e311c2960294aaa0beb5"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.19"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "f923f9a774fcf3f5cb761bfa43aeadd689714813"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.1+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.JLFzf]]
deps = ["REPL", "Random", "fzf_jll"]
git-tree-sha1 = "82f7acdc599b65e0f8ccd270ffa1467c21cb647b"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.11"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "0533e564aae234aff59ab625543145446d8b6ec2"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.1"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "b3ad4a0255688dcb895a52fafbaae3023b588a90"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.4.0"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6893345fd6658c8e475d40155789f4860ac3b21"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.4+0"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "059aabebaa7c82ccb853dd4a0ee9d17796f7e1bc"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.3+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aaafe88dccbd957a8d82f7d05be9b69172e0cee3"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.0.1+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eb62a3deb62fc6d8822c0c4bef73e4412419c5d8"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.8+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c602b1127f4751facb671441ca72715cc95938a"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.3+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "Ghostscript_jll", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "44f93c47f9cd6c7e431f2f2091fcba8f01cd7e8f"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.10"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"
    TectonicExt = "tectonic_jll"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"
    tectonic_jll = "d7dd28d6-a5e6-559c-9131-7eb760cdacc5"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.15.0+0"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "OpenSSL_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.3+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c8da7e6a91781c41a863611c7e966098d783c57a"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.4.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "97bbca976196f2a1eb9607131cb108c69ec3f8a6"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.41.3+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "f04133fe05eff1667d2054c53d59f9122383fe05"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.2+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d0205286d9eceadc518742860bf23f703779a3d6"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.41.3+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f00544d95982ea270145636c181ceda21c4e2575"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.2.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ff69a2b1330bcb730b9ac1ab7dd680176f5896b8"
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.1010+0"

[[deps.Measures]]
git-tree-sha1 = "b513cedd20d9c914783d8ad83d08120702bf2c77"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.3"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.11.4"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6aa4566bb7ae78498a5e68943863fa8b5231b59"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.6+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "NetworkOptions", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "1d1aaa7d449b58415f97d2839c318b70ffb525a0"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.6.1"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.4+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e2bb57a313a74b8104064b7efd01406c0a50d2ff"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.6.1+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.44.0+1"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0662b083e11420952f2e62e17eddae7fc07d5997"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.57.0+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "db76b1ecd5e9715f3d043cec13b2ec93ce015d53"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.44.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.1"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "41031ef3a1be6f5bbbf3e8073f210556daeae5ca"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.3.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "26ca162858917496748aad52bb5d3be4d26a228a"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.4"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "1cc8ad0762e59e713ee3ef28f9b78b2c9f4ca078"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.41.5"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "07a921781cab75691315adc645096ed5e370cb77"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.3"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "522f093a29b31a93e34eaea17ba055d850edea28"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "1d36ef11a9aaf1e8b74dacc6a731dd1de8fd493d"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.3.0"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "34f7e5d2861083ec7596af8b8c092531facf2192"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.8.2+2"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll"]
git-tree-sha1 = "da7adf145cce0d44e892626e647f9dcbe9cb3e10"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.8.2+1"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "9eca9fc3fe515d619ce004c83c31ffd3f85c7ccf"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.8.2+1"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "8f528b0851b5b7025032818eb5abbeb8a736f853"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.8.2+2"

[[deps.REPL]]
deps = ["InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "64d974c2e6fdf07f8155b5b2ca2ffa9069b608d9"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.2"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.12.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "5acc6a41b3082920f79ca3c759acbcecf18a8d78"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.7.1"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "4f96c596b8c8258cc7d3b19797854d368f243ddc"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.4"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "178ed29fd5b2a2cfc3bd31c13375ae925623ff36"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.8.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "aceda6f4e598d331548e04cc6b2124a6148138e3"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.10"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "28145feabf717c5d65c1d5e09747ee7b1ff3ed13"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.6.3"

    [deps.StructUtils.extensions]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.8.3+2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "96478df35bbc2f3e1e791bc7a3d0eeee559e60e9"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.24.0+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "9cce64c0fdd1960b597ba7ecda2950b5ed957438"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.2+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a3ea76ee3f4facd7a64684f9af25310825ee3668"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.2+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "9c7ad99c629a44f81e7799eb05ec2746abb5d588"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.6+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "808090ede1d41644447dd5cbafced4731c56bd2f"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.13+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c74ca84bbabc18c4547014765d194ff0b4dc9da"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.4+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "1a4a26870bf1e5d26cd585e38038d399d7e65706"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.8+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "75e00946e43621e09d431d9b95818ee751e6b2ef"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.2+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "a376af5c7ae60d29825164db40787f15c80c7c54"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.3+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "0ba01bc7396896a4ace8aab67db31403c71628f4"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.7+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c174ef70c96c76f4c3f4d3cfbe09d018bcd1b53"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.6+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "ed756a03e95fff88d8f738ebc2849431bdd4fd1a"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.2.0+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "9750dc53819eba4e9a20be42349a6d3b86c7cdf8"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.6+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f4fc02e384b74418679983a97385644b67e1263b"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll"]
git-tree-sha1 = "68da27247e7d8d8dafd1fcf0c3654ad6506f5f97"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "44ec54b0e2acd408b0fb361e1e9244c60c9c3dd4"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "5b0263b6d080716a02544c55fdff2c8d7f9a16a0"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.10+0"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f233c83cad1fa0e70b7771e0e21b061a116f2763"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.2+0"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "801a858fc9fb90c11ffddee1801bb06a738bda9b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.7+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "00af7ebdc563c9217ecc67776d1bbf037dbcebf4"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.44.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c3b0e6196d50eab0c5ed34021aaa0bb463489510"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.14+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6a34e0e0960190ac2a4363a1bd003504772d631"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.61.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "371cc681c00a3ccc3fbc5c0fb91f58ba9bec1ecf"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.13.1+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "125eedcb0a4a0bba65b657251ce1d27c8714e9d6"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.17.4+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "56d643b57b188d30cccc25e331d416d3d358e557"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.13.4+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "646634dd19587a56ee2f1199563ec056c5f228df"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.4+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "91d05d7f4a9f67205bd6cf395e488009fe85b499"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.28.1+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e015f211ebb898c8180887012b938f3851e719ac"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.55+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll"]
git-tree-sha1 = "11e1772e7f3cc987e9d3de991dd4f6b2602663a5"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.8+0"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b4d631fd51f2e9cdd93724ae25b2efc198b059b1"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.7+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.7.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "14cc7083fc6dff3cc44f2bc435ee96d06ed79aa7"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "10164.0.1+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e7b67590c14d487e734dcb925924c5dc43ec85f3"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "4.1.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "a1fc6507a40bf504527d0d4067d718f8e179b2b8"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.13.0+0"
"""

# ╔═╡ Cell order:
# ╟─952498f6-0aaa-11f1-b51c-39d3a1e83ab9
# ╟─488686f6-3781-43ac-b5b9-2366934e69ec
# ╟─fa395782-bee0-4c61-b649-8ccbca40cca1
# ╠═8498e1f6-6c9b-486f-9b68-7295b119e080
# ╠═f9e57c29-bc78-4e8d-b4f7-3167ea54bc4b
# ╠═ecf1d95f-9994-4551-b287-44ad9458b5d3
# ╠═eba1273a-5100-4a13-b7cd-72e619732231
# ╠═c179b6ac-06aa-46c0-8be1-8044c6bc807f
# ╟─212a44cf-ec9e-4952-a065-60194be0106f
# ╠═58d88ea1-1f6d-4f56-998a-e94c709943ba
# ╟─8a5692af-e265-429c-96ab-f78dd409a113
# ╟─e310d6b8-ed68-49db-aa36-9d2040fed799
# ╟─3f464c20-78b5-46b5-87cc-14e405a694b6
# ╟─47f69b9f-2118-42ba-8d36-b84f0750c31f
# ╟─1e466e15-50cc-43c2-be8c-739967f5d424
# ╠═f912e938-1270-4911-9eef-94c857ab0193
# ╠═c4c36151-4781-480b-9113-43e43fbf20fe
# ╟─b4a97d3f-1d28-4fb3-9a18-82c2df721bfb
# ╠═40343cef-fd41-4215-884e-bd0cc9eac711
# ╠═315f6658-bd33-404c-8d01-55888e41bcab
# ╟─0e496bad-a1d1-4c49-8c00-1b74691453b9
# ╟─8cb8a9f3-ef47-4a77-b1c7-2cfacadd15c8
# ╠═237b7b00-98b3-4339-8ebc-a8861ff4b582
# ╟─bdb29c97-e992-441d-b8c0-72d037c01e96
# ╠═a6633370-4ddc-41af-85f8-b92f88f1e06d
# ╟─3872e2e2-792d-4a98-a025-205493aa381c
# ╠═b09b0a80-ce1e-466c-96f2-7855a9c0abb1
# ╠═4dd4bb10-9d10-402b-a781-c4b6c3e01187
# ╟─e546ec33-ee20-4e6e-a492-ef74d9f9338a
# ╠═df724218-18d0-4192-9b4e-1e440ae846f9
# ╟─1bbaf624-9896-43c5-b316-008598b935b4
# ╟─54cfa7e3-9e2b-4455-820d-eb3afe9229e1
# ╠═da903750-48b5-43d2-8070-a9fb9d70707c
# ╟─f16d89e9-d659-4ee7-b5b6-dbfd58a19e7a
# ╠═e0bc890b-b30b-4820-bb3e-810739cb6e00
# ╟─d5e02d2e-2ca2-41c6-b888-55efcebc0e4c
# ╠═706f9016-275a-4c03-8877-cdfca8353292
# ╟─64d33191-69e1-447c-b91a-2870d6bb26cb
# ╠═90df6819-e288-443f-99a1-796e96de85b7
# ╟─b58646f4-35ba-40bc-aad6-9a0f054fc7cb
# ╟─0f387faf-3ac3-48de-ab1f-68f6e943d11d
# ╠═2a029c1d-ec2f-4731-b35d-59207a75ebe6
# ╟─7177aa90-3646-45aa-8ba9-e3158b9352b7
# ╟─44c295c4-4f6f-4ff8-9115-76264b2fdf9b
# ╟─302e368d-d282-4cbb-946a-8e4f13048bf2
# ╟─d159a679-66cb-41d3-a864-5380cfb870bb
# ╠═f353a3f3-fa2d-4280-a746-a6e2f2e208c3
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
