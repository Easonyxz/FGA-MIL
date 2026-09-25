"""FGA-MIL model."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = dim**-0.5
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(2, dim=-1, keepdim=True) * self.scale
        return x / (norm + self.eps) * self.gamma + self.beta


class AgentAttention(nn.Module):
    """Cross-attention in which agents query WSI instances."""

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.q_norm = AdaptiveNorm(dim)
        self.kv_norm = AdaptiveNorm(dim)
        self.to_q = nn.Linear(dim, dim)
        self.to_kv = nn.Linear(dim, dim * 2)
        self.to_out = nn.Linear(dim, dim)

    def forward(
        self, agents: torch.Tensor, instances: torch.Tensor
    ) -> torch.Tensor:
        q = self.to_q(self.q_norm(agents))
        q = q.reshape(-1, self.num_heads, self.head_dim).transpose(0, 1)

        k, v = self.to_kv(self.kv_norm(instances)).chunk(2, dim=-1)
        k = k.reshape(-1, self.num_heads, self.head_dim).transpose(0, 1)
        v = v.reshape(-1, self.num_heads, self.head_dim).transpose(0, 1)

        attention = (q @ k.transpose(-2, -1) * self.scale).softmax(dim=-1)
        output = (attention @ v).transpose(0, 1).reshape(-1, agents.shape[-1])
        return self.to_out(output)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, time_emb_dim: int, hidden_dim: int):
        super().__init__()
        if time_emb_dim < 4 or time_emb_dim % 2 != 0:
            raise ValueError("time_emb_dim must be an even integer greater than or equal to 4")
        self.time_emb_dim = time_emb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.time_emb_dim // 2
        frequencies = torch.exp(
            torch.arange(half_dim, device=t.device, dtype=t.dtype)
            * (-math.log(10000) / (half_dim - 1))
        )
        embedding = torch.cat(((t * frequencies).sin(), (t * frequencies).cos()), dim=-1)
        return self.time_mlp(embedding)


class AgentBlock(nn.Module):
    def __init__(
        self, dim: int, num_heads: int = 8, ff_mult: int = 2, dropout: float = 0.0
    ):
        super().__init__()
        self.attn = AgentAttention(dim, num_heads)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, agents: torch.Tensor, instances: torch.Tensor
    ) -> torch.Tensor:
        agents = agents + self.attn(agents, instances)
        return agents + self.ffn(agents)


class GatedPooling(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 128):
        super().__init__()
        self.attention_V = nn.Sequential(nn.Linear(dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(dim, hidden_dim), nn.Sigmoid())
        self.attention_w = nn.Linear(hidden_dim, 1)

    def forward(self, agents: torch.Tensor) -> torch.Tensor:
        hidden = self.attention_V(agents) * self.attention_U(agents)
        weights = self.attention_w(hidden).softmax(dim=0)
        return (weights * agents).sum(dim=0)


class FGAMIL(nn.Module):
    """Flow-Guided Agent Multiple Instance Learning (FGA-MIL)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        time_emb_dim: int = 64,
        num_agents: int = 16,
        num_heads: int = 8,
        num_archetypes: int = 8,
        num_agent_blocks: int = 1,
        ff_mult: int = 2,
        use_gated_pooling: bool = True,
        use_inter_class_margin: bool = True,
        inter_class_margin: float = 1.0,
        inter_class_margin_weight: float = 0.1,
        archetype_diversity_loss_weight: float = 0.05,
        dropout_rate: float = 0.1,
        flow_loss_weight: float = 0.15,
        use_class_conditional_flow: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.archetype_diversity_loss_weight = archetype_diversity_loss_weight
        self.use_gated_pooling = use_gated_pooling
        self.use_inter_class_margin = use_inter_class_margin
        self.inter_class_margin = inter_class_margin
        self.inter_class_margin_weight = inter_class_margin_weight
        self.use_flow = flow_loss_weight > 0
        self.use_class_conditional_flow = use_class_conditional_flow and self.use_flow

        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            AdaptiveNorm(hidden_dim),
        )
        self.agents = nn.Parameter(torch.randn(num_agents, hidden_dim))
        self.agent_blocks = nn.ModuleList(
            AgentBlock(hidden_dim, num_heads, ff_mult, dropout_rate)
            for _ in range(num_agent_blocks)
        )
        if use_gated_pooling:
            self.pooling = GatedPooling(hidden_dim, hidden_dim // 2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        )

        self.class_archetypes = nn.Parameter(
            torch.randn(num_classes, num_archetypes, hidden_dim)
        )
        if self.use_flow:
            self.time_embedder = SinusoidalTimeEmbedding(time_emb_dim, hidden_dim)
            if self.use_class_conditional_flow:
                self.class_embedder = nn.Embedding(num_classes, hidden_dim)
                flow_input_dim = hidden_dim * 3
            else:
                self.class_embedder = None
                flow_input_dim = hidden_dim * 2
            self.flow_net = nn.Sequential(
                nn.Linear(flow_input_dim, hidden_dim * 2),
                nn.GELU(),
                nn.LayerNorm(hidden_dim * 2),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )

    def get_archetype_diversity_loss(self) -> torch.Tensor:
        normalized = F.normalize(self.class_archetypes, p=2, dim=-1)
        losses = [torch.exp(-torch.pdist(archetypes).mean()) for archetypes in normalized]
        return torch.stack(losses).mean() * self.archetype_diversity_loss_weight

    def get_inter_class_margin_loss(self) -> torch.Tensor:
        if not self.use_inter_class_margin or self.num_classes < 2:
            return self.class_archetypes.new_zeros(())

        centroids = F.normalize(self.class_archetypes.mean(dim=1), dim=-1)
        similarities = centroids @ centroids.T
        mask = ~torch.eye(
            self.num_classes, dtype=torch.bool, device=similarities.device
        )
        return (
            F.relu(similarities[mask] + self.inter_class_margin).mean()
            * self.inter_class_margin_weight
        )

    def get_auxiliary_losses(self) -> dict[str, torch.Tensor]:
        diversity = self.get_archetype_diversity_loss()
        margin = self.get_inter_class_margin_loss()
        return {"diversity": diversity, "margin": margin, "total": diversity + margin}

    @torch.no_grad()
    def sinkhorn(
        self, cost_matrix: torch.Tensor, epsilon: float = 0.1, n_iters: int = 7
    ) -> torch.Tensor:
        num_agents, num_archetypes = cost_matrix.shape
        log_mu = torch.full(
            (num_agents,), -math.log(num_agents), device=cost_matrix.device
        )
        log_nu = torch.full(
            (num_archetypes,), -math.log(num_archetypes), device=cost_matrix.device
        )
        log_alpha = torch.zeros_like(log_mu)
        for _ in range(n_iters):
            log_beta = log_nu - torch.logsumexp(
                -cost_matrix / epsilon + log_alpha.unsqueeze(-1), dim=0
            )
            log_alpha = log_mu - torch.logsumexp(
                -cost_matrix / epsilon + log_beta.unsqueeze(0), dim=1
            )
        return torch.exp(
            log_alpha.unsqueeze(-1)
            + log_beta.unsqueeze(0)
            - cost_matrix / epsilon
        )

    def _classify(self, agents: torch.Tensor) -> torch.Tensor:
        bag = self.pooling(agents) if self.use_gated_pooling else agents.mean(dim=0)
        return self.classifier(bag)

    def forward(
        self, feature_bags: list[torch.Tensor], labels: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        batch_logits = []
        flow_predictions = []
        flow_targets = []

        for index, features in enumerate(feature_bags):
            instances = self.feature_encoder(features)
            agents = self.agents
            for block in self.agent_blocks:
                agents = block(agents, instances)
            batch_logits.append(self._classify(agents))

            if not (self.training and self.use_flow and labels is not None):
                continue

            label = labels[index].long()
            archetypes = self.class_archetypes[label]
            transport = self.sinkhorn(torch.cdist(agents, archetypes))
            transport = transport / transport.sum(dim=1, keepdim=True).clamp_min(1e-8)
            targets = transport @ archetypes

            t = torch.rand(agents.shape[0], 1, device=agents.device)
            angle = math.pi / 2 * t
            interpolated = angle.cos() * agents + angle.sin() * targets
            target_velocity = math.pi / 2 * (angle.cos() * targets - angle.sin() * agents)

            flow_inputs = [interpolated, self.time_embedder(t)]
            if self.use_class_conditional_flow:
                class_embedding = self.class_embedder(label).expand(agents.shape[0], -1)
                flow_inputs.append(class_embedding)
            flow_predictions.append(self.flow_net(torch.cat(flow_inputs, dim=-1)))
            flow_targets.append(target_velocity)

        logits = torch.stack(batch_logits)
        if flow_predictions:
            return logits, torch.cat(flow_predictions), torch.cat(flow_targets)
        return logits, None, None
