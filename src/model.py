"""
Attention-DragonNet: a doubly-robust neural uplift model with a
self-attention trunk over tokenized tabular features.

Base idea (Dragonnet, Shi/Blei/Veitch 2019): one shared representation feeds
three heads -- propensity, and two potential-outcome heads (treated /
control) -- and a *targeted regularization* term nudges the outcome
predictions using the propensity score, giving a doubly-robust-style
correction that's trained end-to-end rather than applied as a separate
post-hoc step.

Extension here: instead of a plain MLP trunk, each input feature is first
projected into its own embedding ("tokenized"), and a self-attention layer
lets features attend to each other before being pooled into the shared
representation -- similar in spirit to FT-Transformer / TabTransformer for
tabular data. This lets the model learn feature *interactions* (e.g.
loyalty_tier x price_sensitivity, which is exactly what drives the true
treatment effect in our simulated data) rather than relying on a tree's
implicit split-based interactions or a plain MLP's entangled linear mixing.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


CONTINUOUS_FEATURES = [
    "price_sensitivity", "lead_time_days", "past_bookings",
    "length_of_stay", "base_price",
]
CATEGORICAL_FEATURES = {
    "loyalty_tier": 4,   # 4 tiers: 0-3
    "device_mobile": 2,  # binary
}
ALL_FEATURES = CONTINUOUS_FEATURES + list(CATEGORICAL_FEATURES.keys())


@dataclass
class FeatureScaler:
    """Mean/std for continuous features AND the outcome, fit on training data only.

    Standardizing the outcome (booking value, roughly tens to hundreds of
    euros) is what keeps the neural net's loss well-scaled and training
    stable -- without it, the MSE term dwarfs the propensity BCE term and
    the model effectively ignores the propensity/targeted-regularization
    signal. Predictions are un-standardized back to real units downstream.
    """
    means: dict
    stds: dict
    y_mean: float = 0.0
    y_std: float = 1.0

    @classmethod
    def fit(cls, df):
        means, stds = {}, {}
        for c in CONTINUOUS_FEATURES:
            means[c] = float(df[c].mean())
            stds[c] = float(df[c].std() + 1e-8)
        y_mean = float(df["y"].mean())
        y_std = float(df["y"].std() + 1e-8)
        return cls(means=means, stds=stds, y_mean=y_mean, y_std=y_std)

    def transform(self, df):
        out = df.copy()
        for c in CONTINUOUS_FEATURES:
            out[c] = (out[c] - self.means[c]) / self.stds[c]
        return out

    def scale_y(self, y):
        return (y - self.y_mean) / self.y_std

    def unscale_y(self, y_scaled):
        return y_scaled * self.y_std + self.y_mean

    def unscale_cate(self, cate_scaled):
        # A difference of two standardized outcomes only needs re-scaling
        # by std -- the mean cancels out (y1_scaled - y0_scaled) * std.
        return cate_scaled * self.y_std

    def to_dict(self):
        return {"means": self.means, "stds": self.stds, "y_mean": self.y_mean, "y_std": self.y_std}

    @classmethod
    def from_dict(cls, d):
        return cls(means=d["means"], stds=d["stds"], y_mean=d.get("y_mean", 0.0), y_std=d.get("y_std", 1.0))


class AttentionDragonNet(nn.Module):
    """Feature-tokenizing self-attention trunk + Dragonnet-style heads."""

    def __init__(self, d_model: int = 32, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.d_model = d_model

        # One learnable linear projection (token) per continuous feature.
        self.cont_tokenizers = nn.ModuleDict(
            {f: nn.Linear(1, d_model) for f in CONTINUOUS_FEATURES}
        )
        # One embedding table per categorical feature.
        self.cat_embeddings = nn.ModuleDict(
            {f: nn.Embedding(n_cat, d_model) for f, n_cat in CATEGORICAL_FEATURES.items()}
        )
        # A CLS-style summary token, pooled after attention into the shared
        # representation -- analogous to BERT's [CLS].
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=0.1, batch_first=True, activation="gelu",
        )
        # enable_nested_tensor=False deliberately disables PyTorch's
        # "fastpath" / nested-tensor optimization for TransformerEncoder.
        # That fastpath auto-activates under exactly our usage pattern
        # (batch_first=True, eval mode, no attention mask) and has multiple
        # known segfault reports on macOS specifically. Disabling it only
        # changes which internal code path runs the same computation --
        # it doesn't add/remove parameters, so saved weights stay valid.
        self.attention_trunk = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )

        # Dragonnet heads, all reading off the shared (CLS) representation.
        self.propensity_head = nn.Sequential(nn.Linear(d_model, 1))
        self.y0_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, 1))
        self.y1_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, 1))

        # Targeted regularization epsilon (Shi et al. 2019) -- a single
        # learnable scalar used to apply the doubly-robust correction.
        self.epsilon = nn.Parameter(torch.zeros(1))

    def encode(self, x_cont: torch.Tensor, x_cat: dict) -> torch.Tensor:
        """x_cont: (B, n_continuous). x_cat: {feature_name: (B,) long tensor}."""
        batch_size = x_cont.shape[0]
        tokens = []
        for i, f in enumerate(CONTINUOUS_FEATURES):
            tokens.append(self.cont_tokenizers[f](x_cont[:, i : i + 1]))  # (B, d_model)
        for f in CATEGORICAL_FEATURES:
            tokens.append(self.cat_embeddings[f](x_cat[f]))  # (B, d_model)
        tokens = torch.stack(tokens, dim=1)  # (B, n_features, d_model)

        cls = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, d_model)
        seq = torch.cat([cls, tokens], dim=1)  # (B, 1+n_features, d_model)

        encoded = self.attention_trunk(seq)
        return encoded[:, 0, :]  # pooled CLS representation, (B, d_model)

    def forward(self, x_cont: torch.Tensor, x_cat: dict):
        z = self.encode(x_cont, x_cat)
        propensity_logit = self.propensity_head(z).squeeze(-1)
        y0_pred = self.y0_head(z).squeeze(-1)
        y1_pred = self.y1_head(z).squeeze(-1)
        return propensity_logit, y0_pred, y1_pred


def dragonnet_loss(propensity_logit, y0_pred, y1_pred, epsilon, y_true, treatment,
                    alpha: float = 1.0, beta: float = 1.0):
    """Combined loss: outcome MSE + propensity BCE + targeted regularization.

    The targeted regularization term is the piece that gives this a
    doubly-robust flavor: it perturbs the factual outcome prediction using
    the (clipped) inverse-propensity-weighted residual direction `h`, and
    penalizes how far that perturbed prediction is from the true outcome --
    training the network to be self-correcting in exactly the direction
    IPW-style debiasing would push it.
    """
    propensity = torch.sigmoid(propensity_logit)
    propensity_clipped = propensity.clamp(0.01, 0.99)

    y_pred_factual = treatment * y1_pred + (1 - treatment) * y0_pred
    outcome_loss = torch.mean((y_true - y_pred_factual) ** 2)

    propensity_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        propensity_logit, treatment
    )

    h = treatment / propensity_clipped - (1 - treatment) / (1 - propensity_clipped)
    y_pert = y_pred_factual + epsilon * h
    targeted_reg_loss = torch.mean((y_true - y_pert) ** 2)

    total = outcome_loss + alpha * propensity_loss + beta * targeted_reg_loss
    return total, {
        "outcome_loss": outcome_loss.item(),
        "propensity_loss": propensity_loss.item(),
        "targeted_reg_loss": targeted_reg_loss.item(),
    }


def df_to_tensors(df, scaler: FeatureScaler):
    """Convert a covariate DataFrame into the (x_cont, x_cat) tensors the model expects."""
    scaled = scaler.transform(df)
    x_cont = torch.tensor(scaled[CONTINUOUS_FEATURES].values, dtype=torch.float32)
    x_cat = {
        f: torch.tensor(df[f].values, dtype=torch.long) for f in CATEGORICAL_FEATURES
    }
    return x_cont, x_cat


@torch.no_grad()
def predict_cate(model: AttentionDragonNet, df, scaler: FeatureScaler) -> np.ndarray:
    """Predicted CATE = E[Y1|X] - E[Y0|X], read directly off both outcome heads.

    Model outputs are in standardized-outcome units (see FeatureScaler); this
    un-scales the difference back to real booking-value units.
    """
    model.eval()
    x_cont, x_cat = df_to_tensors(df, scaler)
    _, y0_pred, y1_pred = model(x_cont, x_cat)
    cate_scaled = (y1_pred - y0_pred).numpy()
    return scaler.unscale_cate(cate_scaled)


@torch.no_grad()
def predict_propensity(model: AttentionDragonNet, df, scaler: FeatureScaler) -> np.ndarray:
    model.eval()
    x_cont, x_cat = df_to_tensors(df, scaler)
    logit, _, _ = model(x_cont, x_cat)
    return torch.sigmoid(logit).numpy()
