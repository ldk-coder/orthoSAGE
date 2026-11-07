"""Standalone model definition for OrthoSARA DualRAG-V8.

This file intentionally contains model/inference code only:

    base OrthoSARA (anat_a2)
        -> input-side CaseRAG and V8 fusion
        -> output-side frozen GraphRAG refinement

It contains no dataset, index construction, training loop, loss, evaluator, or
command-line code. Retrieval is deliberately an input to the model so a caller
can use any leakage-safe vector index without coupling it to this module.

Tensor conventions
------------------
src_points: [B, T, N, 3], normalized tooth point clouds
tooth_ids:  [B, T], CLIK/Universal-like tooth identifiers
mask:       [B, T], valid-tooth mask
retrieved:  K retrieved training cases, tooth-matched to the query

The output-side GraphRAG is a frozen prediction-only refiner. Its artifact is
loaded by ``OutputGraphRAG.from_artifact``; target/label tensors are not part of
its public API.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
# Geometry utilities
# ---------------------------------------------------------------------------


def apply_se3(points: torch.Tensor, rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    """Apply batched row-vector SE(3)."""
    return torch.matmul(points, rotation.transpose(-1, -2)) + translation.unsqueeze(-2)


def rot6d_to_rotmat(value: torch.Tensor) -> torch.Tensor:
    """Continuous 6D rotation representation to a proper rotation matrix."""
    first = F.normalize(value[..., :3], dim=-1)
    second_raw = value[..., 3:6]
    second = F.normalize(second_raw - (second_raw * first).sum(-1, keepdim=True) * first, dim=-1)
    third = torch.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


def _skew(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(
        vector.shape[:-1] + (3, 3)
    )


def rotvec_to_rotmat(rotvec: torch.Tensor) -> torch.Tensor:
    """Stable Rodrigues exponential map."""
    angle_sq = rotvec.square().sum(dim=-1, keepdim=True)
    angle = torch.sqrt(angle_sq.clamp_min(1.0e-16))
    a = torch.where(
        angle_sq > 1.0e-8,
        torch.sin(angle) / angle,
        1.0 - angle_sq / 6.0 + angle_sq.square() / 120.0,
    )
    b = torch.where(
        angle_sq > 1.0e-8,
        (1.0 - torch.cos(angle)) / angle_sq.clamp_min(1.0e-16),
        0.5 - angle_sq / 24.0 + angle_sq.square() / 720.0,
    )
    skew = _skew(rotvec)
    identity = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device).expand(skew.shape)
    return identity + a.unsqueeze(-1) * skew + b.unsqueeze(-1) * (skew @ skew)


def rotation_to_rotvec(rotation: torch.Tensor) -> torch.Tensor:
    trace = rotation[..., 0, 0] + rotation[..., 1, 1] + rotation[..., 2, 2]
    cosine = torch.clamp((trace - 1.0) * 0.5, -1.0 + 1.0e-6, 1.0 - 1.0e-6)
    angle = torch.acos(cosine)
    vee = torch.stack(
        (
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ),
        dim=-1,
    )
    scale = torch.where(
        angle < 1.0e-4,
        0.5 + angle.square() / 12.0,
        angle / (2.0 * torch.sin(angle).clamp_min(1.0e-6)),
    )
    return vee * scale.unsqueeze(-1)


def project_to_rotation(matrix: torch.Tensor) -> torch.Tensor:
    """Project a batched 3x3 matrix to the closest proper rotation."""
    u, _, vh = torch.linalg.svd(matrix)
    raw = u @ vh
    sign = torch.where(
        torch.linalg.det(raw) < 0,
        -torch.ones((), dtype=matrix.dtype, device=matrix.device),
        torch.ones((), dtype=matrix.dtype, device=matrix.device),
    )
    correction = torch.diag_embed(
        torch.stack((torch.ones_like(sign), torch.ones_like(sign), sign), dim=-1)
    )
    return u @ correction @ vh


# ---------------------------------------------------------------------------
# Tooth graph used by the base model and output GraphRAG
# ---------------------------------------------------------------------------


UPPER_ORDER: tuple[int, ...] = tuple(range(2, 16))
LOWER_ORDER: tuple[int, ...] = tuple(range(18, 32))
SUPPORTED_TOOTH_IDS: tuple[int, ...] = UPPER_ORDER + LOWER_ORDER

INCISORS = frozenset((7, 8, 9, 10, 23, 24, 25, 26))
CANINES = frozenset((6, 11, 22, 27))
PREMOLARS = frozenset((4, 5, 12, 13, 20, 21, 28, 29))
MOLARS = frozenset((2, 3, 14, 15, 18, 19, 30, 31))


def _lookup_tensor(ids: torch.Tensor, mapping: Mapping[int, int], default: int = 0) -> torch.Tensor:
    result = torch.full_like(ids, default)
    for key, value in mapping.items():
        result = torch.where(ids == key, torch.full_like(result, value), result)
    return result


def tooth_type_from_ids(tooth_ids: torch.Tensor) -> torch.Tensor:
    mapping = {tooth_id: 1 for tooth_id in INCISORS}
    mapping.update({tooth_id: 2 for tooth_id in CANINES})
    mapping.update({tooth_id: 3 for tooth_id in PREMOLARS})
    mapping.update({tooth_id: 4 for tooth_id in MOLARS})
    return _lookup_tensor(tooth_ids, mapping)


def tooth_jaw_from_ids(tooth_ids: torch.Tensor) -> torch.Tensor:
    mapping = {tooth_id: 1 for tooth_id in UPPER_ORDER}
    mapping.update({tooth_id: 2 for tooth_id in LOWER_ORDER})
    return _lookup_tensor(tooth_ids, mapping)


def tooth_order_index(tooth_ids: torch.Tensor) -> torch.Tensor:
    mapping = {tooth_id: index for index, tooth_id in enumerate(UPPER_ORDER)}
    mapping.update({tooth_id: 13 - index for index, tooth_id in enumerate(LOWER_ORDER)})
    return _lookup_tensor(tooth_ids, mapping, default=-1)


def tooth_mirror_id(tooth_ids: torch.Tensor) -> torch.Tensor:
    mapping = {tooth_id: UPPER_ORDER[-1 - index] for index, tooth_id in enumerate(UPPER_ORDER)}
    mapping.update({tooth_id: LOWER_ORDER[-1 - index] for index, tooth_id in enumerate(LOWER_ORDER)})
    return _lookup_tensor(tooth_ids, mapping, default=-1)


def clinical_edge_types(
    tooth_ids: torch.Tensor,
    mask: torch.Tensor,
    cross_window: int = 2,
    use_same_type: bool = True,
) -> torch.Tensor:
    """Build edge labels: 1 adjacent, 2 occlusal, 3 mirror, 4 same type."""
    batch_size, teeth = tooth_ids.shape
    valid = mask & (tooth_ids > 0)
    left_id, right_id = tooth_ids[:, :, None], tooth_ids[:, None, :]
    edge_valid = valid[:, :, None] & valid[:, None, :] & (left_id != right_id)

    jaw = tooth_jaw_from_ids(tooth_ids)
    left_jaw, right_jaw = jaw[:, :, None], jaw[:, None, :]
    same_jaw = (left_jaw == right_jaw) & (left_jaw > 0)
    cross_jaw = (left_jaw > 0) & (right_jaw > 0) & (left_jaw != right_jaw)

    order = tooth_order_index(tooth_ids)
    left_order, right_order = order[:, :, None], order[:, None, :]
    order_known = (left_order >= 0) & (right_order >= 0)
    difference = (left_order - right_order).abs()
    mirror = tooth_mirror_id(tooth_ids)[:, :, None]

    adjacent = same_jaw & order_known & (difference == 1)
    occlusal = cross_jaw & order_known & (difference <= int(cross_window))
    mirrored = (mirror == right_id) & same_jaw
    edge_type = torch.zeros((batch_size, teeth, teeth), dtype=torch.long, device=tooth_ids.device)

    if use_same_type:
        types = tooth_type_from_ids(tooth_ids)
        same_type = (
            same_jaw
            & order_known
            & (difference <= 2)
            & (types[:, :, None] == types[:, None, :])
            & (types[:, :, None] > 0)
        )
        edge_type = torch.where(same_type & edge_valid, torch.full_like(edge_type, 4), edge_type)
    edge_type = torch.where(mirrored & edge_valid, torch.full_like(edge_type, 3), edge_type)
    edge_type = torch.where(occlusal & edge_valid, torch.full_like(edge_type, 2), edge_type)
    edge_type = torch.where(adjacent & edge_valid, torch.full_like(edge_type, 1), edge_type)
    return edge_type


def adjacent_pairs(tooth_ids: torch.Tensor, mask: torch.Tensor) -> list[tuple[int, int]]:
    id_to_index = {
        int(tooth_ids[index].item()): int(index)
        for index in torch.where(mask)[0].tolist()
    }
    pairs: list[tuple[int, int]] = []
    for order in (UPPER_ORDER, LOWER_ORDER):
        indices = [id_to_index[tooth_id] for tooth_id in order if tooth_id in id_to_index]
        pairs.extend((left, right) for left, right in zip(indices[:-1], indices[1:]))
    return pairs


# ---------------------------------------------------------------------------
# Base OrthoSARA: anat_a2 branch used by DualRAG-V8
# ---------------------------------------------------------------------------


class ToothShapeEncoder(nn.Module):
    def __init__(self, out_dim: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 64), nn.GELU(),
            nn.Linear(64, 128), nn.GELU(),
            nn.Linear(128, out_dim), nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        self.post = nn.Sequential(nn.LayerNorm(out_dim), nn.Linear(out_dim, out_dim), nn.GELU())

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.post(self.mlp(points).max(dim=1).values)


class M0TokenBuilder(nn.Module):
    def __init__(
        self,
        feat_dim: int = 256,
        id_emb_dim: int = 64,
        d_model: int = 384,
        dropout: float = 0.1,
        max_tooth_id: int = 64,
    ) -> None:
        super().__init__()
        self.id_embedding = nn.Embedding(max_tooth_id + 2, id_emb_dim, padding_idx=0)
        self.proj = nn.Sequential(
            nn.Linear(feat_dim + id_emb_dim + 6, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        shape_feat: torch.Tensor,
        tooth_ids: torch.Tensor,
        mask: torch.Tensor,
        centroid: torch.Tensor,
        rel_centroid: torch.Tensor,
    ) -> torch.Tensor:
        ids = tooth_ids.clamp(min=0, max=self.id_embedding.num_embeddings - 1) + 1
        ids = torch.where(mask, ids, torch.zeros_like(ids))
        token = self.proj(
            torch.cat((shape_feat, self.id_embedding(ids), centroid, rel_centroid), dim=-1)
        )
        return torch.where(mask.unsqueeze(-1), token, torch.zeros_like(token))


class RelationAwareTransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 384,
        nhead: int = 8,
        dropout: float = 0.1,
        init_gate: float = -1.5,
    ) -> None:
        super().__init__()
        self.nhead = nhead
        self.gate = nn.Parameter(torch.tensor(float(init_gate)))
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.rel_bias = nn.Embedding(5, nhead, padding_idx=0)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(
        self,
        value: torch.Tensor,
        edge_type: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, teeth, _ = value.shape
        bias = self.rel_bias(edge_type.clamp(0, 4)).permute(0, 3, 1, 2)
        attention_mask = torch.sigmoid(self.gate) * bias.reshape(
            batch_size * self.nhead, teeth, teeth
        )
        normalized = self.norm1(value)
        attended, _ = self.self_attn(
            normalized,
            normalized,
            normalized,
            attn_mask=attention_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        value = value + self.drop1(attended)
        value = value + self.drop2(self.ffn(self.norm2(value)))
        if key_padding_mask is not None:
            value = torch.where(
                (~key_padding_mask).unsqueeze(-1), value, torch.zeros_like(value)
            )
        return value


class ClinicalRelationTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 384,
        nhead: int = 8,
        depth: int = 6,
        dropout: float = 0.1,
        init_gate: float = -1.5,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            RelationAwareTransformerLayer(d_model, nhead, dropout, init_gate)
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        value: torch.Tensor,
        edge_type: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            value = layer(value, edge_type=edge_type, key_padding_mask=key_padding_mask)
        return self.norm(value)


class PoseDecoder(nn.Module):
    def __init__(self, d_model: int = 384, dropout: float = 0.1) -> None:
        super().__init__()
        self.rot_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, 6),
        )
        self.trans_context = nn.Sequential(
            nn.Linear(6, d_model), nn.LayerNorm(d_model), nn.GELU()
        )
        self.trans_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, 3),
        )

    def forward(
        self, hidden: torch.Tensor, centroid: torch.Tensor, rel_centroid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rotation_6d = self.rot_head(hidden)
        position = self.trans_context(torch.cat((centroid, rel_centroid), dim=-1))
        translation = self.trans_head(hidden + position)
        return rotation_6d, translation


class ClinicalRefinementLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 384,
        rel_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.rel_emb = nn.Embedding(5, rel_dim, padding_idx=0)
        self.msg_mlp = nn.Sequential(
            nn.LayerNorm(d_model + rel_dim + 4),
            nn.Linear(d_model + rel_dim + 4, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model), nn.GELU(),
        )
        self.update = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        center: torch.Tensor,
        edge_type: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, teeth, channels = hidden.shape
        edge_mask = (edge_type > 0) & mask[:, :, None] & mask[:, None, :]
        neighbor_hidden = hidden[:, None, :, :].expand(batch_size, teeth, teeth, channels)
        relative = center[:, None, :, :] - center[:, :, None, :]
        distance = torch.linalg.norm(relative, dim=-1, keepdim=True)
        message = self.msg_mlp(
            torch.cat(
                (neighbor_hidden, self.rel_emb(edge_type.clamp(0, 4)), relative, distance),
                dim=-1,
            )
        )
        relation = edge_type.unsqueeze(-1)
        relation_weight = torch.zeros_like(distance)
        relation_weight = torch.where(relation == 1, torch.ones_like(relation_weight), relation_weight)
        relation_weight = torch.where(relation == 2, torch.full_like(relation_weight, 0.85), relation_weight)
        relation_weight = torch.where(relation == 3, torch.full_like(relation_weight, 0.70), relation_weight)
        relation_weight = torch.where(relation == 4, torch.full_like(relation_weight, 0.35), relation_weight)
        distance_weight = 1.0 / (1.0 + distance.clamp_min(0.0))
        edge_weight = edge_mask.unsqueeze(-1).to(message.dtype) * relation_weight * distance_weight
        aggregate = (message * edge_weight).sum(dim=2) / edge_weight.sum(dim=2).clamp_min(1.0)
        output = hidden + self.update(torch.cat((hidden, aggregate), dim=-1))
        return torch.where(mask.unsqueeze(-1), output, hidden)


class ClinicalPoseRefiner(nn.Module):
    def __init__(
        self,
        d_model: int = 384,
        hidden: int = 256,
        layers: int = 2,
        dropout: float = 0.1,
        cross_window: int = 2,
        max_delta_deg: float = 3.0,
        max_delta_trans: float = 0.02,
        init_gate: float = -1.2,
    ) -> None:
        super().__init__()
        self.cross_window = int(cross_window)
        self.max_delta_rad = math.radians(float(max_delta_deg))
        self.max_delta_trans = float(max_delta_trans)
        self.use_occlusion_features = False
        self.gate = nn.Parameter(torch.tensor(float(init_gate)))
        self.type_emb = nn.Embedding(5, d_model)
        self.pose_proj = nn.Sequential(
            nn.LayerNorm(15), nn.Linear(15, hidden), nn.GELU(), nn.Linear(hidden, d_model)
        )
        self.layers = nn.ModuleList(
            ClinicalRefinementLayer(d_model, 32, dropout) for _ in range(layers)
        )
        self.out = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 6),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        tooth_ids: torch.Tensor,
        tooth_type: torch.Tensor,
        mask: torch.Tensor,
        centroid: torch.Tensor,
        rel_centroid: torch.Tensor,
        base_rotation: torch.Tensor,
        base_translation: torch.Tensor,
        base_points: torch.Tensor,
        edge_type: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        base_center = base_points.mean(dim=2)
        base_move = base_center - centroid
        if edge_type is None:
            edge_type = clinical_edge_types(
                tooth_ids, mask, cross_window=self.cross_window, use_same_type=True
            )
        pose_feature = torch.cat(
            (centroid, rel_centroid, base_center, base_move, base_translation), dim=-1
        )
        value = hidden + self.type_emb(tooth_type.clamp(0, 4)) + self.pose_proj(pose_feature)
        for layer in self.layers:
            value = layer(value, base_center, edge_type, mask)
        raw = self.out(value)
        gate = torch.sigmoid(self.gate)
        delta_rotvec = torch.tanh(raw[..., :3]) * self.max_delta_rad * gate
        delta_translation = torch.tanh(raw[..., 3:]) * self.max_delta_trans * gate
        delta_rotation = rotvec_to_rotmat(delta_rotvec)
        rotation = delta_rotation @ base_rotation
        translation = base_translation + delta_translation
        rotation = torch.where(mask[..., None, None], rotation, base_rotation)
        translation = torch.where(mask[..., None], translation, base_translation)
        return rotation, translation, {
            "base_pred_points": base_points,
            "m1_delta_t_norm": torch.linalg.norm(delta_translation, dim=-1),
            "m1_delta_angle_deg": torch.linalg.norm(delta_rotvec, dim=-1) * 180.0 / math.pi,
            "m1_gate": gate.detach().reshape(1).expand(hidden.shape[0]),
        }


class M0PlusAligner(nn.Module):
    """The anat_a2 OrthoSARA backbone used by the released DualRAG-V8."""

    def __init__(
        self,
        model_variant: str = "anat_a2",
        encoder_variant: str = "pointnet",
        feat_dim: int = 256,
        id_emb_dim: int = 64,
        d_model: int = 384,
        nhead: int = 8,
        depth: int = 6,
        dropout: float = 0.1,
        max_tooth_id: int = 64,
        max_delta_deg: float = 3.0,
        max_delta_trans: float = 0.02,
        cig_layers: int = 2,
        cig_cross_window: int = 2,
        cig_gate_init: float = -1.2,
        **_: Any,
    ) -> None:
        super().__init__()
        if model_variant != "anat_a2":
            raise ValueError("This model-only release contains the anat_a2 branch used by DualRAG-V8")
        if encoder_variant != "pointnet":
            raise ValueError("The released DualRAG-V8 checkpoint uses encoder_variant='pointnet'")
        self.model_variant = model_variant
        self.use_hierarch_encoder = False
        self.use_arch_frame = False
        self.encoder_variant = encoder_variant
        self.use_relation_backbone = True
        self.relation_cross_window = int(cig_cross_window)
        self.shape_encoder = ToothShapeEncoder(feat_dim)
        self.token_builder = M0TokenBuilder(feat_dim, id_emb_dim, d_model, dropout, max_tooth_id)
        self.backbone = ClinicalRelationTransformer(d_model, nhead, depth, dropout)
        self.pose_decoder = PoseDecoder(d_model, dropout)
        self.m1_refiner = ClinicalPoseRefiner(
            d_model=d_model,
            hidden=256,
            layers=cig_layers,
            dropout=dropout,
            cross_window=cig_cross_window,
            max_delta_deg=max_delta_deg,
            max_delta_trans=max_delta_trans,
            init_gate=cig_gate_init,
        )
        # Kept as explicit attributes to document the exact active branch.
        self.cig = None
        self.pose_denoiser = None
        self.center_head = None
        self.type_adapter = None
        self.multi_hyp = None

    def forward(
        self, src_points: torch.Tensor, tooth_ids: torch.Tensor, mask: torch.Tensor
    ) -> dict[str, Any]:
        batch_size, teeth, points, _ = src_points.shape
        centroid = src_points.mean(dim=2)
        local = src_points - centroid.unsqueeze(2)
        shape = self.shape_encoder(local.reshape(batch_size * teeth, points, 3)).reshape(
            batch_size, teeth, -1
        )
        denominator = mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        case_center = (centroid * mask.float().unsqueeze(-1)).sum(
            dim=1, keepdim=True
        ) / denominator.unsqueeze(-1)
        relative_centroid = centroid - case_center
        token = self.token_builder(shape, tooth_ids, mask, centroid, relative_centroid)
        edge_type = clinical_edge_types(
            tooth_ids,
            mask,
            cross_window=self.relation_cross_window,
            use_same_type=True,
        )
        hidden = self.backbone(token, edge_type=edge_type, key_padding_mask=~mask)
        rotation_6d, translation_0 = self.pose_decoder(hidden, centroid, relative_centroid)
        rotation_0 = rot6d_to_rotmat(rotation_6d)
        tooth_type = tooth_type_from_ids(tooth_ids)
        base_points = apply_se3(src_points, rotation_0, translation_0)
        rotation, translation, refine_aux = self.m1_refiner(
            hidden,
            tooth_ids,
            tooth_type,
            mask,
            centroid,
            relative_centroid,
            rotation_0,
            translation_0,
            base_points,
            edge_type,
        )
        return {
            "R": rotation,
            "t_norm": translation,
            "pred_points": apply_se3(src_points, rotation, translation),
            "R0": rotation_0,
            "t0_norm": translation_0,
            "h": hidden,
            "centroid": centroid,
            "rel_centroid": relative_centroid,
            "tooth_type": tooth_type,
            "rel_attn_gate": torch.stack(
                [torch.sigmoid(layer.gate).detach() for layer in self.backbone.layers]
            ).mean().reshape(1).expand(batch_size),
            **refine_aux,
        }


class ArchFormerAligner(nn.Module):
    """Checkpoint-compatible wrapper name used by the original base model."""

    def __init__(self, model_variant: str = "anat_a2", **kwargs: Any) -> None:
        super().__init__()
        self.impl = M0PlusAligner(model_variant=model_variant, **kwargs)
        self.model_variant = model_variant

    def forward(
        self, src_points: torch.Tensor, tooth_ids: torch.Tensor, mask: torch.Tensor
    ) -> dict[str, Any]:
        return self.impl(src_points, tooth_ids, mask)


# ---------------------------------------------------------------------------
# Input CaseRAG and V8 fusion
# ---------------------------------------------------------------------------


@dataclass
class RetrievedCaseBatch:
    """Tooth-matched training-case tensors supplied by a leakage-safe retriever."""

    src_points: torch.Tensor       # [B, K, T, P, 3]
    dst_points: torch.Tensor       # [B, K, T, P, 3]
    rotations: torch.Tensor        # [B, K, T, 3, 3]
    translations: torch.Tensor     # [B, K, T, 3]
    present: torch.Tensor          # [B, K, T]
    scores: torch.Tensor           # [B, K]
    identities: list[list[str]] | None = None
    trace: list[dict[str, Any]] | None = None

    def to(self, device: torch.device | str) -> "RetrievedCaseBatch":
        return RetrievedCaseBatch(
            src_points=self.src_points.to(device),
            dst_points=self.dst_points.to(device),
            rotations=self.rotations.to(device),
            translations=self.translations.to(device),
            present=self.present.to(device),
            scores=self.scores.to(device),
            identities=self.identities,
            trace=self.trace,
        )


class ExpertPairEncoder(nn.Module):
    def __init__(self, d_model: int, max_tooth_id: int, dropout: float) -> None:
        super().__init__()
        self.point_encoder = nn.Sequential(
            nn.Linear(3, 64), nn.GELU(), nn.Linear(64, 128), nn.GELU()
        )
        self.tooth_embedding = nn.Embedding(max_tooth_id + 2, 64, padding_idx=0)
        self.project = nn.Sequential(
            nn.Linear(128 * 2 + 13 + 64, d_model),
            nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self, retrieved: RetrievedCaseBatch, query_tooth_ids: torch.Tensor
    ) -> torch.Tensor:
        source, target = retrieved.src_points, retrieved.dst_points
        source_center, target_center = source.mean(dim=-2), target.mean(dim=-2)
        source_shape = self.point_encoder(source - source_center.unsqueeze(-2)).amax(dim=-2)
        target_shape = self.point_encoder(target - target_center.unsqueeze(-2)).amax(dim=-2)
        rotation_6d = retrieved.rotations[..., :, :2].reshape(
            *retrieved.rotations.shape[:-2], 6
        )
        score = retrieved.scores[:, :, None, None].expand(*source_center.shape[:-1], 1)
        motion = torch.cat(
            (target_center - source_center, retrieved.translations, rotation_6d, score), dim=-1
        )
        ids = query_tooth_ids.clamp_min(-1) + 1
        ids = ids[:, None, :].expand(source.shape[0], source.shape[1], source.shape[2])
        tooth = self.tooth_embedding(ids.clamp(0, self.tooth_embedding.num_embeddings - 1))
        token = self.project(torch.cat((source_shape, target_shape, motion, tooth), dim=-1))
        return torch.where(retrieved.present.unsqueeze(-1), token, torch.zeros_like(token))


class QueryFallbackEncoder(nn.Module):
    def __init__(self, d_model: int, max_tooth_id: int) -> None:
        super().__init__()
        self.point = nn.Sequential(
            nn.Linear(3, 64), nn.GELU(), nn.Linear(64, 128), nn.GELU()
        )
        self.tooth = nn.Embedding(max_tooth_id + 2, 64, padding_idx=0)
        self.project = nn.Sequential(
            nn.Linear(128 + 64 + 3, d_model), nn.LayerNorm(d_model), nn.GELU()
        )

    def forward(
        self, source: torch.Tensor, tooth_ids: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        center = source.mean(dim=-2)
        shape = self.point(source - center.unsqueeze(-2)).amax(dim=-2)
        ids = torch.where(mask, tooth_ids.clamp_min(0) + 1, torch.zeros_like(tooth_ids))
        token = self.project(torch.cat((shape, self.tooth(ids), center), dim=-1))
        return torch.where(mask.unsqueeze(-1), token, torch.zeros_like(token))


class LongHorizonCaseRAGOrthoSARA(nn.Module):
    """Input CaseRAG + V8 fusion used by the released DualRAG-V8."""

    def __init__(
        self,
        base_model: nn.Module,
        *,
        d_model: int = 384,
        nhead: int = 8,
        dropout: float = 0.1,
        max_tooth_id: int = 64,
        max_rag_gate: float = 0.48,
        retrieval_score_temperature: float = 20.0,
        max_residual_rotation_deg: float = 2.0,
        max_residual_translation_norm: float = 0.02,
        max_residual_gate: float = 0.10,
        retrieval_dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.rag_gate_logit = nn.Parameter(torch.tensor(0.0))
        self.base_model = base_model
        self.d_model = int(d_model)
        self.max_prior_blend = float(max_rag_gate)
        self.retrieval_score_temperature = float(retrieval_score_temperature)
        self.max_rag_rotation_rad = math.radians(float(max_residual_rotation_deg))
        self.max_rag_translation_norm = float(max_residual_translation_norm)
        self.max_residual_gate = float(max_residual_gate)
        self.retrieval_dropout = float(retrieval_dropout)
        self.rag_strength = 0.0

        self.expert_encoder = ExpertPairEncoder(d_model, max_tooth_id, dropout)
        self.query_fallback = QueryFallbackEncoder(d_model, max_tooth_id)
        self.query_projection = nn.Linear(d_model, d_model)
        nn.init.eye_(self.query_projection.weight)
        nn.init.zeros_(self.query_projection.bias)
        self.cross_attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.pose_delta = nn.Linear(d_model, 6)
        nn.init.zeros_(self.pose_delta.weight)
        nn.init.zeros_(self.pose_delta.bias)
        self.tooth_gate = nn.Linear(d_model * 2, 1)
        nn.init.zeros_(self.tooth_gate.weight)
        nn.init.constant_(self.tooth_gate.bias, -2.0)
        self.pose_context = nn.Sequential(
            nn.Linear(12, d_model), nn.LayerNorm(d_model), nn.GELU()
        )
        self.prior_gate = nn.Linear(d_model * 2 + 4, 1)
        nn.init.zeros_(self.prior_gate.weight)
        nn.init.zeros_(self.prior_gate.bias)

    def set_rag_strength(self, value: float) -> None:
        self.rag_strength = float(max(0.0, min(1.0, value)))

    def _consensus_pose(
        self,
        retrieved: RetrievedCaseBatch,
        base_rotation: torch.Tensor,
        base_translation: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        present = retrieved.present & mask[:, None, :]
        logits = retrieved.scores[:, :, None].expand_as(present).to(base_translation.dtype)
        logits = (logits * self.retrieval_score_temperature).masked_fill(~present, -1.0e4)
        weights = torch.softmax(logits, dim=1) * present.to(logits.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        translation = (weights.unsqueeze(-1) * retrieved.translations).sum(dim=1)
        rotation_matrix = (
            weights.unsqueeze(-1).unsqueeze(-1) * retrieved.rotations
        ).sum(dim=1)
        rotation = project_to_rotation(rotation_matrix).detach()
        has_prior = present.any(dim=1)
        translation = torch.where(has_prior.unsqueeze(-1), translation, base_translation)
        rotation = torch.where(
            has_prior.unsqueeze(-1).unsqueeze(-1), rotation, base_rotation
        )
        return rotation, translation, has_prior, weights

    def forward(
        self,
        src_points: torch.Tensor,
        tooth_ids: torch.Tensor,
        mask: torch.Tensor,
        retrieved: RetrievedCaseBatch,
    ) -> dict[str, Any]:
        base = self.base_model(src_points, tooth_ids, mask)
        base_rotation = base["R"]
        base_translation = base["t_norm"]
        base_points = base["pred_points"]
        base_pivot = base_points.mean(dim=-2)

        consensus_rotation, consensus_translation, has_prior, consensus_weights = (
            self._consensus_pose(retrieved, base_rotation, base_translation, mask)
        )
        base_rotation_6d = base_rotation[..., :, :2].reshape(
            *base_rotation.shape[:-2], 6
        )
        pose_context = self.pose_context(
            torch.cat((base_rotation_6d, base_translation, base_pivot), dim=-1)
        )
        base_hidden = base.get("h")
        query = (
            self.query_fallback(src_points, tooth_ids, mask)
            if base_hidden is None
            else self.query_projection(base_hidden.detach())
        ) + pose_context

        expert = self.expert_encoder(retrieved, tooth_ids)
        expert_mask = retrieved.present.clone()
        if self.training and self.retrieval_dropout > 0:
            keep = torch.rand(expert_mask.shape[:2], device=expert_mask.device) >= self.retrieval_dropout
            keep[:, 0] = True
            expert_mask = expert_mask & keep.unsqueeze(-1)
        confidence = consensus_weights * float(consensus_weights.shape[1])
        expert = torch.where(
            expert_mask.unsqueeze(-1), expert * confidence.unsqueeze(-1), torch.zeros_like(expert)
        )

        batch_size, ranks, teeth, channels = expert.shape
        memory = expert.permute(0, 2, 1, 3).reshape(batch_size * teeth, ranks, channels)
        valid = expert_mask.permute(0, 2, 1).reshape(batch_size * teeth, ranks)
        no_match = ~valid.any(dim=1)
        if bool(no_match.any()):
            memory = memory.clone()
            valid = valid.clone()
            memory[no_match, 0] = 0.0
            valid[no_match, 0] = True
        attended, attention_weights = self.cross_attention(
            query.reshape(batch_size * teeth, 1, channels),
            memory,
            memory,
            key_padding_mask=~valid,
            need_weights=True,
            average_attn_weights=False,
        )
        attended = attended.reshape(batch_size, teeth, channels)
        attended = torch.where(
            no_match.reshape(batch_size, teeth, 1), torch.zeros_like(attended), attended
        )

        prior_blend = torch.where(
            mask & has_prior,
            torch.full_like(
                base_translation[..., 0],
                self.max_prior_blend * 0.5 * float(self.rag_strength),
            ),
            torch.zeros_like(base_translation[..., 0]),
        )
        relative_rotation = consensus_rotation @ base_rotation.transpose(-1, -2)
        relative_rotvec = rotation_to_rotvec(relative_rotation)
        prior_rotation = rotvec_to_rotmat(
            relative_rotvec * prior_blend.unsqueeze(-1)
        ) @ base_rotation
        prior_translation = (
            (1.0 - prior_blend.unsqueeze(-1)) * base_translation
            + prior_blend.unsqueeze(-1) * consensus_translation
        )
        prior_points = apply_se3(src_points, prior_rotation, prior_translation)

        fused = self.norm1(query + attended)
        fused = self.norm2(fused + self.feed_forward(fused))
        raw_delta = self.pose_delta(fused)
        tooth_gate = torch.sigmoid(
            self.tooth_gate(torch.cat((query, attended), dim=-1))
        ).squeeze(-1)
        global_gate = torch.sigmoid(self.rag_gate_logit)
        raw_residual_gate = tooth_gate * global_gate * float(self.rag_strength)
        residual_gate = torch.clamp(raw_residual_gate, max=self.max_residual_gate)
        residual_gate = torch.where(mask, residual_gate, torch.zeros_like(residual_gate))
        rotvec = (
            torch.tanh(raw_delta[..., :3])
            * self.max_rag_rotation_rad
            * residual_gate.unsqueeze(-1)
        )
        delta_translation = (
            torch.tanh(raw_delta[..., 3:])
            * self.max_rag_translation_norm
            * residual_gate.unsqueeze(-1)
        )
        delta_rotation = rotvec_to_rotmat(rotvec)
        pivot = prior_points.mean(dim=-2)
        rotation = delta_rotation @ prior_rotation
        translation = (
            torch.matmul(delta_rotation, (prior_translation - pivot).unsqueeze(-1)).squeeze(-1)
            + pivot
            + delta_translation
        )
        prediction = apply_se3(src_points, rotation, translation)
        return {
            **base,
            "base_R": base_rotation,
            "base_t_norm": base_translation,
            "base_pred_points": base_points,
            "consensus_R": consensus_rotation,
            "consensus_t_norm": consensus_translation,
            "consensus_weights": consensus_weights,
            "prior_R": prior_rotation,
            "prior_t_norm": prior_translation,
            "prior_pred_points": prior_points,
            "prior_blend": prior_blend,
            "R": rotation,
            "t_norm": translation,
            "pred_points": prediction,
            "rag_rotvec": rotvec,
            "rag_delta_t_norm": delta_translation,
            "rag_gate": residual_gate,
            "rag_gate_raw": torch.where(mask, raw_residual_gate, torch.zeros_like(raw_residual_gate)),
            "rag_attention": attention_weights.reshape(
                batch_size, teeth, *attention_weights.shape[1:]
            ),
            "retrieved_identities": retrieved.identities,
            "retrieval_trace": retrieved.trace,
        }


# ---------------------------------------------------------------------------
# Output GraphRAG: frozen SE(3) head + graph-routed collision constraints
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OutputGraphRAGConfig:
    rotation_shrinkage: float = 0.5
    translation_shrinkage: float = 0.5
    max_rotation_delta_deg: float = 2.0
    max_translation_delta_mm: float = 0.08
    min_predicted_rotation_gain_deg: float = 0.0
    min_predicted_translation_gain_mm: float = 0.0
    min_predicted_point_gain_mm: float = 0.0
    collision_enabled: bool = True
    collision_steps: int = 12
    collision_learning_rate: float = 0.025
    collision_weight: float = 1.0
    collision_trust_weight: float = 1.0
    collision_arch_weight: float = 0.15
    collision_max_delta_mm: float = 0.08
    collision_clearance_mm: float = 0.2
    collision_sample_points: int = 256


class FrozenLinearHead(nn.Module):
    """Frozen standardized linear model stored in the GraphRAG artifact."""

    def __init__(
        self,
        feature_mean: Sequence[float],
        feature_scale: Sequence[float],
        coefficients: Sequence[Sequence[float]],
    ) -> None:
        super().__init__()
        self.register_buffer("feature_mean", torch.tensor(feature_mean, dtype=torch.float64))
        self.register_buffer("feature_scale", torch.tensor(feature_scale, dtype=torch.float64))
        self.register_buffer("coefficients", torch.tensor(coefficients, dtype=torch.float64))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        mean = self.feature_mean.to(device=features.device, dtype=features.dtype)
        scale = self.feature_scale.to(device=features.device, dtype=features.dtype)
        coefficients = self.coefficients.to(device=features.device, dtype=features.dtype)
        standardized = (features - mean) / scale
        design = torch.cat((standardized, torch.ones_like(standardized[..., :1])), dim=-1)
        return design @ coefficients


def _cap_vectors(vectors: torch.Tensor, cap: float) -> torch.Tensor:
    norm = torch.linalg.norm(vectors, dim=-1, keepdim=True).clamp_min(1.0e-12)
    return vectors * torch.clamp(float(cap) / norm, max=1.0)


def _sample_points(points: torch.Tensor, maximum: int) -> torch.Tensor:
    if maximum <= 0 or points.shape[-2] <= maximum:
        return points
    index = torch.linspace(0, points.shape[-2] - 1, maximum, device=points.device).long()
    return points.index_select(-2, index)


def output_graph_features(
    points_mm: torch.Tensor,
    tooth_ids: torch.Tensor,
    mask: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
    clearance_mm: float,
) -> torch.Tensor:
    """Prediction-only tooth/node features consumed by the output GraphRAG."""
    centroid = points_mm.mean(dim=-2)
    extent = points_mm.amax(dim=-2) - points_mm.amin(dim=-2)
    case_centroid = centroid[mask].mean(dim=0) if bool(mask.any()) else centroid.new_zeros(3)
    relative = centroid - case_centroid
    one_hot = points_mm.new_zeros((points_mm.shape[0], len(SUPPORTED_TOOTH_IDS)))
    column = {tooth_id: index for index, tooth_id in enumerate(SUPPORTED_TOOTH_IDS)}
    for index in torch.where(mask)[0].tolist():
        position = column.get(int(tooth_ids[index].item()))
        if position is not None:
            one_hot[index, position] = 1.0

    previous_offset = points_mm.new_zeros((points_mm.shape[0], 3))
    next_offset = points_mm.new_zeros((points_mm.shape[0], 3))
    flags = points_mm.new_zeros((points_mm.shape[0], 4))
    sampled = _sample_points(points_mm.detach(), 64)
    id_to_index = {
        int(tooth_ids[index].item()): int(index)
        for index in torch.where(mask)[0].tolist()
    }
    for order in (UPPER_ORDER, LOWER_ORDER):
        available = [id_to_index[tooth_id] for tooth_id in order if tooth_id in id_to_index]
        for position, index in enumerate(available):
            if position > 0:
                previous = available[position - 1]
                previous_offset[index] = centroid[previous] - centroid[index]
                distance = torch.cdist(
                    sampled[index : index + 1], sampled[previous : previous + 1]
                ).amin()
                flags[index, 0] = 1.0
                flags[index, 2] = torch.relu(points_mm.new_tensor(clearance_mm) - distance)
            if position + 1 < len(available):
                following = available[position + 1]
                next_offset[index] = centroid[following] - centroid[index]
                distance = torch.cdist(
                    sampled[index : index + 1], sampled[following : following + 1]
                ).amin()
                flags[index, 1] = 1.0
                flags[index, 3] = torch.relu(points_mm.new_tensor(clearance_mm) - distance)
    features = torch.cat(
        (
            one_hot,
            rotation.reshape(points_mm.shape[0], 9),
            translation,
            centroid,
            relative,
            extent,
            previous_offset,
            next_offset,
            flags,
        ),
        dim=-1,
    )
    return torch.where(mask.unsqueeze(-1), features, torch.zeros_like(features))


class OutputGraphRAG(nn.Module):
    """Frozen, target-blind output GraphRAG inference core.

    The static tooth graph routes two actions: bounded learned SE(3) residuals
    and bounded adjacent-tooth collision correction. The frozen linear heads
    are loaded from the released refiner artifact.
    """

    def __init__(
        self,
        residual_head: FrozenLinearHead,
        quality_gate_head: FrozenLinearHead,
        config: OutputGraphRAGConfig,
    ) -> None:
        super().__init__()
        self.residual_head = residual_head
        self.quality_gate_head = quality_gate_head
        self.config = config
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @classmethod
    def from_artifact(cls, path: str | Path) -> "OutputGraphRAG":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("frozen") is not True:
            raise ValueError("Output GraphRAG requires a frozen artifact")
        config_payload = payload["config"]
        allowed = OutputGraphRAGConfig.__dataclass_fields__.keys()
        config = OutputGraphRAGConfig(
            **{key: config_payload[key] for key in allowed if key in config_payload}
        )
        residual = payload["residual_model"]
        quality = payload["quality_gate_model"]
        return cls(
            FrozenLinearHead(
                residual["feature_mean"], residual["feature_scale"], residual["coefficients"]
            ),
            FrozenLinearHead(
                quality["feature_mean"], quality["feature_scale"], quality["coefficients"]
            ),
            config,
        )

    def _collision_refine(
        self,
        points: torch.Tensor,
        tooth_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        config = self.config
        zero = points.new_zeros((points.shape[0], 3))
        if not config.collision_enabled or config.collision_steps <= 0:
            return points, zero
        pairs = adjacent_pairs(tooth_ids, mask)
        if not pairs:
            return points, zero
        sampled = _sample_points(points.detach(), config.collision_sample_points)

        def pair_penetration(delta: torch.Tensor) -> torch.Tensor:
            index = torch.tensor(pairs, dtype=torch.long, device=points.device)
            left, right = index.unbind(dim=-1)
            left_points = sampled.index_select(0, left) + delta.index_select(0, left).unsqueeze(1)
            right_points = sampled.index_select(0, right) + delta.index_select(0, right).unsqueeze(1)
            distance = torch.cdist(left_points, right_points).amin(dim=(-2, -1))
            return torch.relu(float(config.collision_clearance_mm) - distance)

        baseline = pair_penetration(zero)
        if not bool((baseline > 1.0e-4).any()):
            return points, zero
        # Frozen GraphRAG action-plan routing: the retrieved DTAN collision
        # audit rule supplies a bounded [1, 2] severity multiplier, while the
        # patient graph bounds the translation cap by its worst violation.
        max_violation = float(baseline.max().item())
        violation_ratio = min(
            max(max_violation / max(float(config.collision_clearance_mm), 1.0e-6), 0.0),
            1.0,
        )
        effective_collision_weight = float(config.collision_weight) * (1.0 + violation_ratio)
        effective_max_delta = min(float(config.collision_max_delta_mm), max_violation)
        with torch.enable_grad():
            delta = zero.clone().requires_grad_(True)
            optimizer = torch.optim.Adam([delta], lr=float(config.collision_learning_rate))
            for _ in range(int(config.collision_steps)):
                optimizer.zero_grad(set_to_none=True)
                collision = pair_penetration(delta).mean()
                trust = delta[mask].square().mean() if bool(mask.any()) else delta.sum() * 0.0
                laplacian_terms = []
                id_to_index = {
                    int(tooth_ids[index].item()): int(index)
                    for index in torch.where(mask)[0].tolist()
                }
                for order in (UPPER_ORDER, LOWER_ORDER):
                    indices = [id_to_index[item] for item in order if item in id_to_index]
                    if len(indices) >= 3:
                        arch_delta = delta[indices]
                        laplacian_terms.append(
                            (arch_delta[:-2] - 2.0 * arch_delta[1:-1] + arch_delta[2:]).square().mean()
                        )
                arch = torch.stack(laplacian_terms).mean() if laplacian_terms else delta.sum() * 0.0
                loss = (
                    effective_collision_weight * collision
                    + float(config.collision_trust_weight) * trust
                    + float(config.collision_arch_weight) * arch
                )
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    norm = torch.linalg.norm(delta, dim=-1, keepdim=True).clamp_min(1.0e-12)
                    delta.mul_(torch.clamp(effective_max_delta / norm, max=1.0))
                    delta.mul_(mask.unsqueeze(-1).to(delta.dtype))
            candidate = delta.detach()

        best, best_score = zero, float(baseline.mean().item())
        for scale in (1.0, 0.5, 0.25, 0.125):
            trial = candidate * scale
            score = float(pair_penetration(trial).mean().item())
            if score + 1.0e-9 < best_score:
                best, best_score = trial, score
        return points + best.unsqueeze(-2), best

    def _refine_case(
        self,
        src_points_mm: torch.Tensor,
        points_mm: torch.Tensor,
        tooth_ids: torch.Tensor,
        mask: torch.Tensor,
        rotation: torch.Tensor,
        translation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        config = self.config
        features = output_graph_features(
            points_mm,
            tooth_ids,
            mask,
            rotation,
            translation,
            config.collision_clearance_mm,
        )
        raw = self.residual_head(features)
        rotation_delta = _cap_vectors(
            raw[..., :3] * float(config.rotation_shrinkage),
            math.radians(float(config.max_rotation_delta_deg)),
        )
        translation_delta = _cap_vectors(
            raw[..., 3:] * float(config.translation_shrinkage),
            float(config.max_translation_delta_mm),
        )
        proposed = torch.cat((rotation_delta, translation_delta), dim=-1)
        gate_features = torch.cat(
            (
                features,
                proposed,
                torch.linalg.norm(rotation_delta, dim=-1, keepdim=True),
                torch.linalg.norm(translation_delta, dim=-1, keepdim=True),
            ),
            dim=-1,
        )
        predicted_gain = self.quality_gate_head(gate_features)
        gate = (
            (predicted_gain[..., 0] >= config.min_predicted_rotation_gain_deg)
            & (predicted_gain[..., 1] >= config.min_predicted_translation_gain_mm)
            & (predicted_gain[..., 2] >= config.min_predicted_point_gain_mm)
            & mask
        )
        candidate_rotation = rotvec_to_rotmat(rotation_delta) @ rotation
        candidate_translation = translation + translation_delta
        refined_rotation = torch.where(gate[:, None, None], candidate_rotation, rotation)
        refined_translation = torch.where(gate[:, None], candidate_translation, translation)
        refined_points = apply_se3(src_points_mm, refined_rotation, refined_translation)
        final_points, collision_delta = self._collision_refine(
            refined_points, tooth_ids, mask
        )
        final_translation = refined_translation + collision_delta
        return final_points, refined_rotation, final_translation, {
            "accepted_se3_teeth": int(gate.sum().item()),
            "collision_max_delta_mm": float(
                torch.linalg.norm(collision_delta, dim=-1).max().item()
            ),
            "target_fields_available_to_output_rag": False,
        }

    def forward(
        self,
        *,
        src_points_mm: torch.Tensor,
        pred_points_mm: torch.Tensor,
        tooth_ids: torch.Tensor,
        mask: torch.Tensor,
        rotation: torch.Tensor,
        translation_global: torch.Tensor,
    ) -> dict[str, Any]:
        points_rows, rotation_rows, translation_rows, diagnostics = [], [], [], []
        for batch_index in range(pred_points_mm.shape[0]):
            result = self._refine_case(
                src_points_mm[batch_index],
                pred_points_mm[batch_index],
                tooth_ids[batch_index],
                mask[batch_index],
                rotation[batch_index],
                translation_global[batch_index],
            )
            points_rows.append(result[0])
            rotation_rows.append(result[1])
            translation_rows.append(result[2])
            diagnostics.append(result[3])
        return {
            "pred_points_mm": torch.stack(points_rows),
            "R": torch.stack(rotation_rows),
            "t_global": torch.stack(translation_rows),
            "diagnostics": diagnostics,
            "target_fields_available_to_output_rag": False,
        }


# ---------------------------------------------------------------------------
# End-to-end DualRAG wrapper and checkpoint helpers
# ---------------------------------------------------------------------------


class DualRAGV8(nn.Module):
    """Base OrthoSARA -> CaseRAG/V8 -> output GraphRAG."""

    def __init__(
        self,
        input_case_rag: LongHorizonCaseRAGOrthoSARA,
        output_graph_rag: OutputGraphRAG | None = None,
    ) -> None:
        super().__init__()
        self.input_case_rag = input_case_rag
        self.output_graph_rag = output_graph_rag

    @staticmethod
    def _global_translation(
        rotation: torch.Tensor,
        translation_norm: torch.Tensor,
        case_center: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        rotated_center = torch.matmul(
            case_center[:, None, None, :], rotation.transpose(-1, -2)
        ).squeeze(-2)
        return (
            case_center[:, None, :]
            + scale[:, None, None] * translation_norm
            - rotated_center
        )

    def forward(
        self,
        src_points: torch.Tensor,
        tooth_ids: torch.Tensor,
        mask: torch.Tensor,
        retrieved: RetrievedCaseBatch,
        *,
        case_center: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        input_result = self.input_case_rag(src_points, tooth_ids, mask, retrieved)
        output: dict[str, Any] = {"input_case_rag": input_result}
        if self.output_graph_rag is None:
            return output
        if case_center is None or scale is None:
            raise ValueError("case_center and scale are required by output GraphRAG")
        points_mm = (
            input_result["pred_points"] * scale[:, None, None, None]
            + case_center[:, None, None, :]
        )
        source_mm = (
            src_points * scale[:, None, None, None]
            + case_center[:, None, None, :]
        )
        translation_global = self._global_translation(
            input_result["R"], input_result["t_norm"], case_center, scale
        )
        output["output_graph_rag"] = self.output_graph_rag(
            src_points_mm=source_mm,
            pred_points_mm=points_mm,
            tooth_ids=tooth_ids,
            mask=mask,
            rotation=input_result["R"],
            translation_global=translation_global,
        )
        return output


def build_input_dualrag_v8(
    *,
    max_rag_gate: float = 0.48,
    d_model: int = 384,
    nhead: int = 8,
    dropout: float = 0.1,
    max_tooth_id: int = 64,
) -> LongHorizonCaseRAGOrthoSARA:
    base = ArchFormerAligner(
        model_variant="anat_a2",
        encoder_variant="pointnet",
        feat_dim=256,
        id_emb_dim=64,
        d_model=d_model,
        nhead=nhead,
        depth=6,
        dropout=dropout,
        max_tooth_id=max_tooth_id,
        max_delta_deg=3.0,
        max_delta_trans=0.02,
        cig_layers=2,
        cig_cross_window=2,
        cig_gate_init=-1.2,
    )
    return LongHorizonCaseRAGOrthoSARA(
        base,
        d_model=d_model,
        nhead=nhead,
        dropout=dropout,
        max_tooth_id=max_tooth_id,
        max_rag_gate=max_rag_gate,
    )


def load_v8_weights(
    model: LongHorizonCaseRAGOrthoSARA,
    checkpoint_path: str | Path,
    *,
    use_ema: bool = True,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load released V8 weights; this is inference support, not training code."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(checkpoint["model"], strict=True)
    if use_ema and "ema_parameters" in checkpoint:
        named = dict(model.named_parameters())
        if set(named) != set(checkpoint["ema_parameters"]):
            raise RuntimeError("EMA parameter names do not match DualRAG-V8")
        with torch.no_grad():
            for name, parameter in named.items():
                parameter.copy_(checkpoint["ema_parameters"][name])
    model.set_rag_strength(float(checkpoint.get("rag_strength", 1.0)))
    return checkpoint


__all__ = [
    "ArchFormerAligner",
    "M0PlusAligner",
    "RetrievedCaseBatch",
    "LongHorizonCaseRAGOrthoSARA",
    "OutputGraphRAG",
    "OutputGraphRAGConfig",
    "DualRAGV8",
    "build_input_dualrag_v8",
    "load_v8_weights",
]
