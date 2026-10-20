# Derivations of formulas used in the code

## Sampling from $p(x_t \vert{} x_s, x_0)$

$$x_s = \alpha_s x_0 + \sigma_s \epsilon_1 \\ x_t = \alpha_t x_0 + \sigma_t \epsilon_2 \\ \epsilon_1 = \frac{x_s - \alpha_s x_0}{\sigma_s}$$

Since both noise variables come from the same process, $\epsilon_2$ must be correlated with $\epsilon_1$.

**Lemma:**

Every standard normal variable $\epsilon_2$ that is correlated with another standard normal variable $\epsilon_1$ can be written as a linear combination of $\epsilon_1$ and a completely new, orthogonal (independent) noise $Z \sim \mathcal{N}(0, I)$.

From the additional assumption that $Var(\epsilon_2) = 1$ and $Cov(\epsilon_1, \epsilon_2) = \rho$, it follows that:

$$\epsilon_2 = \rho \epsilon_1 + \sqrt{1-\rho^2} Z$$

Therefore:

$$x_t = \alpha_t x_0 + \sigma_t \left( \rho \epsilon_1 + \sqrt{1-\rho^2} Z \right)$$

$$x_t = \alpha_t x_0 + \sigma_t \rho \left( \frac{x_s - \alpha_s x_0}{\sigma_s} \right) + \sigma_t \sqrt{1-\rho^2} Z$$

$$x_t = \left(\frac{\sigma_t \rho}{\sigma_s}\right) x_s + \left( \alpha_t - \frac{\sigma_t \rho \alpha_s}{\sigma_s} \right) x_0 + \sigma_t \sqrt{1-\rho^2} Z$$

For $\rho = \frac{\alpha_t \sigma_s}{\alpha_s \sigma_t}$, the term with $x_0$ cancels out completely:

$$\alpha_t - \frac{\sigma_t \alpha_s}{\sigma_s} \left( \frac{\alpha_t \sigma_s}{\alpha_s \sigma_t} \right) = \alpha_t - \alpha_t = 0$$

$$x_t = \frac{\alpha_t}{\alpha_s} x_s + \sqrt{\sigma_t^2 - \left(\frac{\alpha_t}{\alpha_s}\right)^2 \sigma_s^2} \, Z$$

Thus $p(x_t \vert{} x_s, x_0) = p(x_t \vert{} x_s)$ for chosen $\rho$.

## Reparametrization of model prediction

Reparametrization equations are derived using the following transition formulas:

$$
\begin{gather*}
        x_0^*(x_t) = {\mathbb{E}[x_0 \vert{} x_t]} \\
        \epsilon_\theta^*(x_t) = \mathbb{E}[\epsilon \vert{} x_t] = \mathbb{E}\left[ \frac{x_t - \alpha_t x_0}{\sigma_t} \Big\vert{} x_t \right] = \frac{x_t}{\sigma_t} - \frac{\alpha_t}{\sigma_t} {\mathbb{E}[x_0 \vert{} x_t]} \\
        v = \alpha_t \epsilon - \sigma_t x_0 = \frac{\alpha_t}{\sigma_t} x_t - \frac{1}{\sigma_t} x_0 \implies v_\theta^*(x_t) = \frac{\alpha_t}{\sigma_t} x_t - \frac{1}{\sigma_t} {\mathbb{E}[x_0 \vert{} x_t]} \\
        {\alpha_t\mathbb{E}[x_0 \vert{} x_t]} = x_t + \sigma_t^2 \nabla_{x_t} \log p_t(x_t) \implies s^*(x_{t}) = \frac{{\alpha_t\mathbb{E}[x_0 \vert{} x_t]} - x_{t}}{\sigma^2_{t}}
\end{gather*}
$$

and the fact that $\nabla_{x_t} \log p_\sigma(\frac{x_t}{\alpha_t} \mid x_0) = \frac{1}{\alpha_t} \nabla_{\tilde{x}} \log p_\sigma(\tilde{x} \mid x_0)$.