using LaTeXStrings
using LinearAlgebra
using Plots

#ldoc on
#=
---
subtitle: Fixed point iteration and Newton
date: 2026-03-27
engine: jupyter
---

::: {.content-hidden unless-format="html"}
{{< include _commonm.tex >}}
:::

# Fixed points and contraction mappings

```{julia}
#|echo: false
#|output: false
include("2026-03-27.jl")
```

::: {.content-hidden unless-format="pdf"}
```{julia}
#|echo: false
#|output: false
pgfplotsx()
```
:::

As discussed in previous lectures, many iterations we consider have the
form
$$
  x^{k+1} = G(x^k)
$$
where $G : \mathbb{R}^n \rightarrow \mathbb{R}^n$. We call $G$ a
*contraction* on $\Omega$ if it is Lipschitz with constant less than
one, i.e.
$$
  \|G(x)-G(y)\| \leq \alpha \|x-y\|, \quad \alpha < 1.
$$
A sufficient (but not necessary) condition for $G$ to be Lipschitz on
$\Omega$ is if $G$ is differentiable and $\|G'(x)\| \leq \alpha$ for
all $x \in \Omega$.

According to the *contraction mapping theorem* or *Banach fixed point
theorem*, when $G$ is a contraction on a closed set $\Omega \subset
\mathbb{R}^n$ and $G(\Omega) \subset \Omega$, there is a unique fixed
point $x^* \in \Omega$ (i.e. a point such that $x^* - G(x^*)$).
The proof is short.
First, note that $\|x^{l+1}-x^l\| \leq \alpha^l \|x^1-x^0\|$.
Then if $x^k$ and $x^m$ are iterates where $m > k$, we can bound
$\|x^m-x^k\|$ via a telescoping series:
$$\begin{aligned}
  \|x^m-x^k\|
  &= \left\| \sum_{l=k}^{m-1} x^{l+1}-x^l \right\| \\
  &\leq \sum_{l=k}^{m-1} \|x^{l+1}-x^l \| \\
  &\leq \sum_{l=k}^{m-1} \alpha^l \|x^1-x^0\| \\
  &\leq \alpha^k \frac{\|x^1-x^0\|}{1-\alpha}.
\end{aligned}$$
Hence, for any $\epsilon > 0$ there is an $N$ such that when $k, m
\geq N$, $\|x^m-x^k\| < \epsilon$.  This means that the sequence is a
*Cauchy sequence*, and therefore it converges to a limiting point
$x^*$.  This point must satisfy the fixed point equation $G(x^*) =
x^*$ by the limiting argument and by continuity of $G$.  Moreover,
there can be no other other fixed points: if $G(z) = z$,
then
$$
  \|z-x^*\| = \|G(z)-G(x^*)\| \leq \alpha \|z-x^*\|;
$$
and the only way this can be true is if $\|z-x^*\| = 0$.

## Uses of contraction mappings

If we can express the solution of a nonlinear equation as the fixed
point of a contraction mapping, we get two immediate benefits.

First, we know that a solution exists and is unique (at least, it is
unique within $\Omega$). This is a nontrivial advantage, as it is easy
to write nonlinear equations that have no solutions, or have continuous
families of solutions, without realizing that there is a problem.

Second, we have a numerical method -- albeit a potentially slow one --
for computing the fixed point. We take the fixed point iteration
$$
  x^{k+1} = G(x^k)
$$
started from some $x^0 \in \Omega$, and we subtract
the fixed point equation $x^* = G(x^*)$ to get an iteration for
$e^k = x^k-x^*$:
$$
  e^{k+1} = G(x^* + e^k) - G(x^*)
$$
Using contractivity, we get
$$
  \|e^{k+1}\| = \|G(x^* + e^k) - G(x^*)\| \leq \alpha\|e^k\|
$$
which implies that $\|e^k\| \leq \alpha^k \|e^0\| \rightarrow 0.$

When error goes down by a factor of $\alpha > 0$ at each step, we say
the iteration is *linearly convergent* (or *geometrically convergent*).
The name reflects a semi-logarithmic plot of (log) error versus
iteration count; if the errors lie on a straight line, we have linear
convergence. Contractive fixed point iterations converge at least
linearly, but may converge more quickly.

## Asymptotic convergence behavior

Consider fixed point iteration $x^{k+1} = G(x^k)$ where $G$ is
a contraction mappign with constant $\alpha < 1$.  By the fixed point
theorem, we have a *bound* on the error $e^k = x^k-x^*$ given by
$$
  \|e^k\| \leq \alpha^k \|e^0\|.
$$
However, the iteration may actually converge rather more quickly than
the bound suggests.  To get a more realistic understanding of convergence,
at least when $G$ is continuously differentiable, it is useful to consider
a Taylor expansion around the fixed point:
$$
  G(x^k) = G(x^* + e^k) \approx G(x^*) + G'(x^*) e^k = x^* + G'(x^*) e^k.
$$
Subtracting $x^*$, we have the iteration
$$
  e^{k+1} = G'(x^*) e^k + O(\|e^k\|^2).
$$
We have analyzed iterations like this before when looking at
eigenvalue problems.  The asymptotic rate of convergence is generally
determined by the spectral radius $\rho(G'(x^*))$, and can be bounded
by any consistent norm of $G'(x^*)$.  In the case when $G'(x^*) = 0$,
the error at step $k+1$ behaves like the square of the error at step $k$,
and we have *quadratic* convergence.

## A toy example

Consider the function $G : \mathbb{R}^2 \rightarrow \mathbb{R}^2$ given
by
$$
  G(x) =
  \frac{1}{4} \begin{bmatrix}
    x_1-\cos(x_2) \\ x_2-\sin(x_1)
  \end{bmatrix}
$$
This is a contraction mapping on all of $\mathbb{R}^2$ (why?).

Let's look at $\|x^k-G(x^k)\|$ as a function of $k$, starting from the
initial guess $x^0 = 0$ (@fig-toy-contraction-cvg).
=#

function test_toy_contraction()

    G(x) = 0.25*[x[1]-cos(x[2]); x[2]-sin(x[1])]

    # Run the fixed point iteration for 100 steps
    resid_norms = []
    x = zeros(2)
    for k = 1:100
        x = G(x)
        push!(resid_norms, norm(x-G(x), Inf))
    end

    x, resid_norms
end

#=
```{julia}
#|label: fig-toy-contraction-cvg
#|fig-cap: "Convergence for `test_toy_contraction`."
#|echo: false
let
    x, resid_norms = test_toy_contraction()
    resid_norms = resid_norms[resid_norms .> 0.0]
    plot(0:length(resid_norms)-1, resid_norms, yscale=:log10,
         xlabel=L"k", ylabel=L"\|x_k-x_{k+1}\|", legend=false)
end
```

## Questions

1.  Show that for the example above $\|G'\|_\infty \leq \frac{1}{2}$
    over all of $\mathbb{R}^2$. This implies that
    $\|G(x)-G(y)\|_\infty \leq \frac{1}{2} \|x-y\|_\infty$.

2.  The mapping $x \mapsto x/2$ is a contraction on $(0, \infty)$, but
    does not have a fixed point on that interval. Why does this not
    contradict the contraction mapping theorem?

3.  For $S > 0$, show that the mapping $g(x) = \frac{1}{2} (x + S/x)$ is
    a contraction on the interval $[\sqrt{S}, \infty)$. What is the
    fixed point? What is the Lipschitz constant?

# Newton's method for nonlinear equations

The idea behind Newton's method is to approximate a nonlinear
$f \in C^1$ by linearizations around successive guesses.  We then get the
next guess by finding where the linearized approximaton is zero.  That
is, we set
$$
  f(x^{k+1}) \approx f(x^k) + f'(x^k) (x^{k+1}-x^k) = 0,
$$
which we can rearrange to
$$
  x^{k+1} = x^k - f'(x^k)^{-1} f(x^k).
$$
Note the implicit assumption here that $f'(x^k)$ is nonsingular!

Of course, we do not actually want to form an inverse, and to set the
stage for later variations on the method, we also write the iteration
as
$$\begin{aligned}
  f'(x^k) p^k &= -f(x^k) \\
  x^{k+1} &= x^k + p^k.
\end{aligned}$$

## Superlinear convergence

Suppose $f(x^*) = 0$.  Taylor expansion about $x^k$ gives
$$
  0 = f(x^*) = f(x^k) + f'(x^k) (x^*-x^k) + r(x^k)
$$
where the remainder term $r(x^k)$ is $o(\|x^k-x^*\|) = o(\|e^k\|)$.  Hence
$$
  x^{k+1} = x^* + f'(x^{k})^{-1} r(x^k)
$$
and subtracting $x^*$ from both sides gives
$$
  e^{k+1} = f'(x^k)^{-1} r(x^k) = f'(x^k)^{-1} o(\|e^k\|)
$$
If $\|f'(x)^{-1}\|$ is bounded for $x$ near $x^*$ and $x^0$ is close
enough, this is sufficient to guarantee *superlinear convergence*.
When we have a stronger condition, such as $f'$ Lipschitz, we get
*quadratic convergence*, i.e. $e^{k+1} = O(\|e^k\|^2)$.  Of course,
this is all local theory -- we need a good initial guess!

## A toy example

Consider the problem of finding the solutions to the system
$$\begin{aligned}
  x + 2y &= 2 \\
  x^2 + 4y^2 &= 4.
\end{aligned}$$
That is, we are looking for the intersection of a straight line and an
ellipse.  This is a simple enough problem that we can compute the
solution in closed form; there are intersections at $(0, 1)$ and at
$(2, 0)$.  Suppose we did not know this, and instead wanted to solve
the system by Newton's iteration.  To do this, we need to write the
problem as finding the zero of some function
$$
  f(x,y) = \begin{bmatrix} x + 2y - 2 \\ x^2 + 4y^2 - 4 \end{bmatrix} = 0.
$$
We also need the Jacobian $J = f'$:
$$
  \frac{\partial f}{\partial (x, y)} =
  \begin{bmatrix}
    \frac{\partial f_1}{\partial x} & \frac{\partial f_1}{\partial y} \\
    \frac{\partial f_2}{\partial x} & \frac{\partial f_2}{\partial y}
  \end{bmatrix}.
$$

We show the convergence of the Newton iteration in @fig-toy-newton-cvg.
=#

function test_toy_newton(x, y)

    # Set up the function and the Jacobian
    f(x) = [x[1] + 2*x[2]-2;
            x[1]^2 + 4*x[2]^2 - 4]
    J(x) = [1      2     ;
            2*x[1] 8*x[2]]

    # Run ten steps of Newton from the initial guess
    x = [x; y]
    fx = f(x)
    resids = zeros(10)
    for k = 1:10
        x -= J(x)\fx
        fx = f(x)
        resids[k] = norm(fx)
    end

    x, resids
end

#=
```{julia}
#|label: fig-toy-newton-cvg
#|fig-cap: "Convergence for `test_toy_newton` to $(0,1)$."
#|echo: false
let
    # Plot the residuals on a semilog scale
    x, resids = test_toy_newton(1.0, 2.0)
    plot(resids[resids .> 0], yscale=:log10,
         ylabel=L"k", xlabel=L"\|f(x^k)\|", legend=false)
end
```

## Questions

1. Finding an (real) eigenvalue of $A$ can be posed as a nonlinear
   equation solving problem: we want to find $x$ and $\lambda$ such that
   $Ax = \lambda x$ and $x^T x = 1$.  Write a Newton iteration for this
   problem.
=#
