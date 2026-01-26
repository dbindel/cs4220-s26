### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ 73972049-bbff-4f49-84b5-f523d850c75f
using LinearAlgebra

# ╔═╡ 79070b07-064d-43af-897e-31ec656d7f3c
using BenchmarkTools

# ╔═╡ 90d28c78-a2e0-45d6-abe8-b246bf7fec1c
using Test

# ╔═╡ a88d628e-facd-11f0-8715-e9ff7f2f7acd
md"""
# HW 1 for CS 4220

You may (and probably should) talk about problems with the each other, with the TAs, and with me, providing attribution for any good ideas you might get. Your final write-up should be your own.
"""

# ╔═╡ 34a57804-c96f-44ab-9e94-f30d4e642871
md"""
## 1. Placing parens

Suppose $A\in \mathbb{R}^{n \times n}$ and $d, u, v, w \in \mathbb{R}^n$.  Rewrite each of the following Julia functions to compute the same result but with the indicated asymptotic complexity.  Your solution codes should take the same arguments, but be named `hw1a_s`, `hw1b_s`, and `hw1c_s`.

**Hint**: `hw1c_s` can still be a one-liner using [`cumsum`](https://docs.julialang.org/en/v1/base/arrays/#Base.cumsum).
"""

# ╔═╡ 698c7b49-d6cc-4d1c-9213-e66feb49de19
# Rewrite to take O(n) time
hw1a(d, u, v) = dot(diagm(d), u*v')

# ╔═╡ 1a3b2b99-9012-449e-b766-812dc72608af
# Rewrite to take O(n^2) time
hw1b(A, d, u, v, w) = A*(I + u*v')*w

# ╔═╡ 6b700f70-7b85-4bef-8caa-882fd76ab26e
# Rewrite to take O(n) time
hw1c(u, v, w) = tril(u*v')*w

# ╔═╡ 2c6a99f7-a916-4db4-b8de-df7cfafe25fb
md"""
You may want to check that your solutions are correct translations with the following code.
"""

# ╔═╡ 98422fdc-ace0-40bf-b763-cab94d373a87
# ╠═╡ disabled = true
#=╠═╡
let
	n = 1000
	A = rand(n,n)
	d, u, v, w = rand(n), rand(n), rand(n), rand(n)

	@test hw1a(d, u, v) ≈ hw1a_s(d, u, v)
	@test hw1b(A, d, u, v, w) ≈ hw1b_s(A, d, u, v, w)
	@test hw1c(u, v, w) ≈ hw1c_s(u, v, w)
end
  ╠═╡ =#

# ╔═╡ 7c9335b9-0a38-4876-a5e5-7c62b0e4de26
md"""
## 2. Chebyshev change of basis

Recall the [first-kind Chebyshev polynomials](https://en.wikipedia.org/wiki/Chebyshev_polynomials) follow the recurrence

$$\begin{aligned}
T_0(x) &= 1 \\
T_1(x) &= x \\
T_{k+1}(x) &= 2x T_k(x) - T_{k-1}(x)
\end{aligned}$$

We have seen in the slides how to write $T_2 = P_2 X$ where $X \in \mathbb{R}^{3 \times 3}$ is an upper triangular matrix representing the change of basis from the power basis $P_2 = \begin{bmatrix} 1 & x & x^2 \end{bmatrix}$ to the Chebyshev basis $T_2 = \begin{bmatrix} T_0(x) & T_1(x) & T_2(x) \end{bmatrix}$.  Write a function `power_to_cheb(d)` that constructs a more general $X \in \mathbb{R}^{(d+1) \times (d+1)}$ such that $T_d = P_d X$.

**Hint**: If $p(x) = P_d(x) c$, then

$$x p(x) = P_{d+1}(x) \begin{bmatrix} 0 \\ c \end{bmatrix}$$
"""

# ╔═╡ 0f938a8f-21d2-4c3d-8b47-8710b37520c9
md"""
You may use the following basic sanity checker to test.
"""

# ╔═╡ 2c2a00cf-d1d5-4a3d-b034-0269bb3fdfad
# ╠═╡ disabled = true
#=╠═╡
@test power_to_cheb(2) == [1.0 0.0 -1.0;
						   0.0 1.0  0.0;
						   0.0 0.0  2.0]
  ╠═╡ =#

# ╔═╡ 57b5084f-d50a-40ee-9558-3f3b87656093
md"""
## 3. Polynomial norms

For $\mathcal{P}_2$ on $[-1,1]$ what are $\|p\|_2$, $\|p\|_1$, and $\|p\|_\infty$ for $p(x) = x$?
"""

# ╔═╡ c17c4162-9684-48d5-957e-87d7ef747474
md"""
## 4. Norm constants

For $v \in \mathbb{R}^n$, argue that

$$\|v\|_\infty \leq \|v\|_2 \leq \sqrt{n} \|v\|_\infty.$$
"""

# ╔═╡ da743cbe-3fd3-4dcd-be79-409a9a630beb
md"""
### Solution

By definition,

$$\|v\|_\infty = \max_j |v_j|$$

Therefore

$$\|v\|_\infty = \max_j |v_j| \leq \sum_{j=1}^n |v_j|^2 \leq \sum_{j=1}^n \|v\|_\infty = n \|v\|_\infty.$$

Taking square roots completes the argument.
"""

# ╔═╡ 2ddbc6bf-cc73-4e84-a538-1e6e21f01d68
md"""
## 5. Consistency of Frobenius norm

Argue that for any $A \in \mathbb{R}^{m \times n}$ and $x \in \mathbb{R}^n$,

$$\|Ax\|_2 \leq \|A\|_F \|x\|_2.$$
"""

# ╔═╡ a1e400c5-0859-449c-8598-4d2a091f814b
md"""
## 6. Quadratic reconstruction

Suppose $\phi : \mathbb{R}^n \rightarrow \mathbb{R}$ is a quadratic form.  Write a function `form_to_matrix(phi, n)` to compute the associated matrix such that $\phi(x) = x^T A x$.  You may want to use the following tester to sanity check.
"""

# ╔═╡ c4efb195-cda6-42ee-b884-d4bed8ede783
# ╠═╡ disabled = true
#=╠═╡
let
	A = rand(10,10)
	A = (A+A')/2
	phi(x) = x'*A*x
	@test form_to_matrix(phi, 10) ≈ A
end
  ╠═╡ =#

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[compat]
BenchmarkTools = "~1.5.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.4"
manifest_format = "2.0"
project_hash = "6f817cfe641be3f0b5a4e153f3d4f77f42847e22"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "f1dff6729bc61f4d49e140da1af55dcd1ac97b2f"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.5.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

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

[[deps.Profile]]
deps = ["StyledStrings"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

    [deps.Statistics.weakdeps]
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"
"""

# ╔═╡ Cell order:
# ╟─a88d628e-facd-11f0-8715-e9ff7f2f7acd
# ╠═73972049-bbff-4f49-84b5-f523d850c75f
# ╠═79070b07-064d-43af-897e-31ec656d7f3c
# ╠═90d28c78-a2e0-45d6-abe8-b246bf7fec1c
# ╟─34a57804-c96f-44ab-9e94-f30d4e642871
# ╠═698c7b49-d6cc-4d1c-9213-e66feb49de19
# ╠═1a3b2b99-9012-449e-b766-812dc72608af
# ╠═6b700f70-7b85-4bef-8caa-882fd76ab26e
# ╟─2c6a99f7-a916-4db4-b8de-df7cfafe25fb
# ╠═98422fdc-ace0-40bf-b763-cab94d373a87
# ╟─7c9335b9-0a38-4876-a5e5-7c62b0e4de26
# ╟─0f938a8f-21d2-4c3d-8b47-8710b37520c9
# ╠═2c2a00cf-d1d5-4a3d-b034-0269bb3fdfad
# ╟─57b5084f-d50a-40ee-9558-3f3b87656093
# ╟─c17c4162-9684-48d5-957e-87d7ef747474
# ╟─da743cbe-3fd3-4dcd-be79-409a9a630beb
# ╟─2ddbc6bf-cc73-4e84-a538-1e6e21f01d68
# ╟─a1e400c5-0859-449c-8598-4d2a091f814b
# ╠═c4efb195-cda6-42ee-b884-d4bed8ede783
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
