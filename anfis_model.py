"""
anfis_model.py
---------------
Implementação de um ANFIS (Adaptive Neuro-Fuzzy Inference System):

Arquitetura em 5 camadas:

  Camada 1 (Fuzzificação): pertinência gaussiana por entrada/termo
  Camada 2 (Regras):       produto (T-norm) das pertinências -> força
                            de disparo de cada regra
  Camada 3 (Normalização): força normalizada de cada regra
  Camada 4 (Consequentes): combinação linear das entradas por regra
                            (Sugeno de 1a ordem)   — TREINADA POR LSE
  Camada 5 (Agregação):    soma ponderada -> saída final (risco)
"""

import itertools
import torch
import torch.nn as nn


class ANFIS(nn.Module):
    def __init__(self, n_inputs: int = 3, n_mfs_per_input: int = 3):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_mfs = n_mfs_per_input
        self.n_rules = n_mfs_per_input ** n_inputs

        # ANTECEDENTES: gaussianas com centro mu e largura sigma.
        # Estes são os únicos parâmetros treinados por gradiente descendente.
        self.mu = nn.Parameter(
            torch.rand(n_inputs, n_mfs_per_input, dtype=torch.float32)
        )
        self.sigma = nn.Parameter(
            torch.ones(n_inputs, n_mfs_per_input, dtype=torch.float32) * 0.3
        )

        # Índices de combinação das regras (produto cartesiano dos termos
        # de cada entrada), definindo a base de m^n regras Sugeno.
        self.rule_indices = list(
            itertools.product(range(n_mfs_per_input), repeat=n_inputs)
        )

        # CONSEQUENTES de Sugeno de 1a ordem: y_r = a_r1*x1 + ... + a_rn*xn + b_r
        # Estes NAO sao nn.Parameter porque sao atualizados por LSE (fora
        # do grafo de autograd), passe direto do Jang. Iniciam em zero e
        # sao atualizados na primeira epoca.
        self.register_buffer(
            "consequents",
            torch.zeros(self.n_rules, n_inputs + 1, dtype=torch.float32),
        )

    # ---------------------------------------------------------------
    def init_membership_from_ranges(self, ranges):
        """Distribui os termos linguisticos uniformemente na faixa observada
        de cada entrada."""
        with torch.no_grad():
            for i in range(self.n_inputs):
                lo, hi = ranges[i]
                span = hi - lo if hi > lo else 1.0
                centers = torch.linspace(
                    lo + 0.15 * span, hi - 0.15 * span, self.n_mfs
                )
                self.mu[i, :] = centers
                self.sigma[i, :] = span / (self.n_mfs * 1.5)

    # ---------------------------------------------------------------
    def _memberships(self, x: torch.Tensor) -> torch.Tensor:
        """Camada 1 - gaussianas."""
        x_exp = x.unsqueeze(-1)                          # (batch, n_inputs, 1)
        mu = self.mu.unsqueeze(0)                        # (1, n_inputs, n_mfs)
        sigma = self.sigma.unsqueeze(0).clamp(min=1e-3)
        return torch.exp(-0.5 * ((x_exp - mu) / sigma) ** 2)

    def _firing_normalized(self, x: torch.Tensor) -> torch.Tensor:
        """Camadas 2 e 3 - forca de disparo produto + normalizacao."""
        batch_size = x.shape[0]
        memberships = self._memberships(x)               # (batch, n_inputs, n_mfs)

        firing_list = []
        for combo in self.rule_indices:
            prod = torch.ones(batch_size, device=x.device)
            for i, term_idx in enumerate(combo):
                prod = prod * memberships[:, i, term_idx]
            firing_list.append(prod)
        firing = torch.stack(firing_list, dim=1)         # (batch, n_rules)

        firing_sum = firing.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return firing / firing_sum

    # ---------------------------------------------------------------
    def _design_matrix(self, x: torch.Tensor,
                        firing_norm: torch.Tensor) -> torch.Tensor:
        """
        Constroi a matriz de projeto A para o problema de minimos quadrados
        do passe direto de Jang. Cada linha corresponde a uma amostra e
        cada coluna a um parametro consequente aplanado.

        Reescrevendo com um vetor de parametros p (n_rules * (n_inputs+1)):
                        y_hat  = A . p
        onde a linha da amostra k tem, para cada regra r, os termos
            w_bar_{k,r} * x_{k,1}, ... , w_bar_{k,r} * x_{k,n}, w_bar_{k,r}
        """
        batch_size = x.shape[0]
        x_bias = torch.cat(
            [x, torch.ones(batch_size, 1, device=x.device)], dim=1
        )  # (batch, n_inputs + 1)
        # Produto externo por amostra: w_bar_r * [x1, x2, ..., xn, 1]
        A = (
            firing_norm.unsqueeze(-1) * x_bias.unsqueeze(1)
        )  # (batch, n_rules, n_inputs+1)
        return A.reshape(batch_size, -1)  # (batch, n_rules * (n_inputs+1))

    # ---------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inferencia: propagacao direta com os consequentes atuais."""
        batch_size = x.shape[0]
        firing_norm = self._firing_normalized(x)         # (batch, n_rules)

        x_bias = torch.cat(
            [x, torch.ones(batch_size, 1, device=x.device)], dim=1
        )
        rule_outputs = x_bias @ self.consequents.T       # (batch, n_rules)
        return (firing_norm * rule_outputs).sum(dim=1)


# ===================================================================
def _lse_update_consequents(model: ANFIS,
                             x_train: torch.Tensor,
                             y_train: torch.Tensor) -> None:
    """
    mantem os antecedentes fixos e resolve os
    consequentes por minimos quadrados. Como o modelo Sugeno e LINEAR
    nos consequentes, LSE da a solucao exata que minimiza a perda MSE
    para essa configuracao de antecedentes.

    Usa torch.linalg.lstsq (decomposicao QR / SVD, numericamente estavel)
    em vez da equacao normal explicita (A^T A)^-1 A^T y.
    """
    with torch.no_grad():
        firing_norm = model._firing_normalized(x_train)              # (N, R)
        A = model._design_matrix(x_train, firing_norm)               # (N, R*(n+1))
        result = torch.linalg.lstsq(A, y_train.unsqueeze(-1))
        p = result.solution.squeeze(-1)                              # (R*(n+1),)
        model.consequents.copy_(
            p.reshape(model.n_rules, model.n_inputs + 1)
        )


def train_anfis_hybrid(model: ANFIS,
                        x_train: torch.Tensor,
                        y_train: torch.Tensor,
                        epochs: int = 200,
                        lr_antecedents: float = 0.01,
                        verbose: bool = True):
    """
      Cada epoca faz:
      (1) Passe direto: LSE para os consequentes (uma etapa exata).
      (2) Passe reverso: um passo de gradiente descendente para os
          antecedentes (mu, sigma) da camada 1.

    Como o LSE da o minimo global do problema linear a cada epoca, o
    numero total de epocas necessarias e uma ordem de grandeza menor
    do que treinar tudo por gradiente.
    """
    # Otimizador SO para os antecedentes; consequentes ficam fora do grafo.
    optimizer = torch.optim.Adam(
        [model.mu, model.sigma], lr=lr_antecedents
    )
    loss_fn = nn.MSELoss()
    history = []

    for epoch in range(epochs):
        # (1) Passe direto - LSE (analitico)
        _lse_update_consequents(model, x_train, y_train)

        # (2) Passe reverso - gradiente descendente nos antecedentes
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        optimizer.step()

        history.append(loss.item())

        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
            print(f"Epoca {epoch:4d} | MSE: {loss.item():.5f}")

    # Uma ultima chamada de LSE com os antecedentes finais, garantindo
    # que os consequentes estao em sincronia com as MFs finais.
    _lse_update_consequents(model, x_train, y_train)
    return history


# Alias de compatibilidade
train_anfis = train_anfis_hybrid
