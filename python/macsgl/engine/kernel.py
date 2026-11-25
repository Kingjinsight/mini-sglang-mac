from __future__ import annotations

import math

import mlx.core as mx

_NUM_GPU_TARGET_GROUPS = 128  # split the context until roughly this many threadgroups are busy
_MAX_PARTITIONS = 16

# One threadgroup per (request, kv head, partition). It serves all G query heads that share the
# kv head, so every K/V row is loaded once. Simdgroups stride over keys, lanes split head_dim.
# Scores live in the log2 domain (q is pre-scaled by log2(e) / sqrt(D)).
_DECODE_SOURCE = """
    constexpr int E = D / 32;
    const uint b = threadgroup_position_in_grid.y / HKV;
    const uint h = threadgroup_position_in_grid.y % HKV;
    const uint p = threadgroup_position_in_grid.z;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;

    const int len = lens[b];
    const int chunk = (len + P - 1) / P;
    const int start = int(p) * chunk;
    const int end = min(start + chunk, len);
    const device int* row = page_table + long(table_idx[b]) * PT_STRIDE;

    const float scale = M_LOG2E_F * rsqrt(float(D));
    float qv[G][E], acc[G][E], m[G], l[G];
    for (int g = 0; g < G; ++g) {
        const device T* qp = q + ((long(b) * HKV + h) * G + g) * D + lane * E;
        for (int i = 0; i < E; ++i) {
            qv[g][i] = float(qp[i]) * scale;
            acc[g][i] = 0.0f;
        }
        m[g] = -INFINITY;
        l[g] = 0.0f;
    }

    for (int j = start + int(sg); j < end; j += NSG) {
        const long base = (long(row[j]) * HKV + h) * D + lane * E;
        float kv[E], vv[E];
        for (int i = 0; i < E; ++i) {
            kv[i] = float(k_pool[base + i]);
            vv[i] = float(v_pool[base + i]);
        }
        for (int g = 0; g < G; ++g) {
            float s = 0.0f;
            for (int i = 0; i < E; ++i) {
                s += qv[g][i] * kv[i];
            }
            s = simd_sum(s);
            const float m_new = max(m[g], s);
            const float alpha = fast::exp2(m[g] - m_new);
            const float w = fast::exp2(s - m_new);
            l[g] = l[g] * alpha + w;
            for (int i = 0; i < E; ++i) {
                acc[g][i] = acc[g][i] * alpha + w * vv[i];
            }
            m[g] = m_new;
        }
    }

    threadgroup float sm_m[NSG][G];
    threadgroup float sm_l[NSG][G];
    threadgroup float sm_acc[NSG][G][D];
    for (int g = 0; g < G; ++g) {
        if (lane == 0) {
            sm_m[sg][g] = m[g];
            sm_l[sg][g] = l[g];
        }
        for (int i = 0; i < E; ++i) {
            sm_acc[sg][g][lane * E + i] = acc[g][i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = thread_index_in_threadgroup; idx < G * D; idx += NSG * 32) {
        const uint g = idx / D, d = idx % D;
        float mx_ = -INFINITY;
        for (int s = 0; s < NSG; ++s) {
            mx_ = max(mx_, sm_m[s][g]);
        }
        float lsum = 0.0f, o = 0.0f;
        if (mx_ != -INFINITY) {  // a partition may hold no keys at all
            for (int s = 0; s < NSG; ++s) {
                const float w = fast::exp2(sm_m[s][g] - mx_);
                lsum += w * sm_l[s][g];
                o += w * sm_acc[s][g][d];
            }
        }
        const long head = (long(b) * HKV + h) * G + g;
        STORE_RESULT
    }
"""
_STORE_FINAL = "out[head * D + d] = T(o / lsum);"
_STORE_PARTIAL = """
        const long part = head * P + p;
        if (d == 0) {
            part_m[part] = mx_;
            part_l[part] = lsum;
        }
        part_o[part * D + d] = o;
"""

_COMBINE_SOURCE = """
    const uint head = threadgroup_position_in_grid.y;
    const uint d = thread_position_in_threadgroup.x;
    float mx_ = -INFINITY;
    for (int p = 0; p < P; ++p) {
        mx_ = max(mx_, part_m[head * P + p]);
    }
    float lsum = 0.0f, o = 0.0f;
    for (int p = 0; p < P; ++p) {
        const float mp = part_m[head * P + p];
        if (mp != -INFINITY) {
            const float w = fast::exp2(mp - mx_);
            lsum += w * part_l[head * P + p];
            o += w * part_o[(long(head) * P + p) * D + d];
        }
    }
    out[long(head) * D + d] = T(o / lsum);
"""

_INPUTS = ["q", "k_pool", "v_pool", "page_table", "table_idx", "lens"]
_decode_single = mx.fast.metal_kernel(
    name="macsgl_paged_decode",
    input_names=_INPUTS,
    output_names=["out"],
    source=_DECODE_SOURCE.replace("STORE_RESULT", _STORE_FINAL),
)
_decode_split = mx.fast.metal_kernel(
    name="macsgl_paged_decode_split",
    input_names=_INPUTS,
    output_names=["part_m", "part_l", "part_o"],
    source=_DECODE_SOURCE.replace("STORE_RESULT", _STORE_PARTIAL),
)
_combine = mx.fast.metal_kernel(
    name="macsgl_paged_decode_combine",
    input_names=["part_m", "part_l", "part_o"],
    output_names=["out"],
    source=_COMBINE_SOURCE,
)


def num_partitions(batch_size: int, num_kv_heads: int) -> int:
    groups = batch_size * num_kv_heads
    return max(1, min(_MAX_PARTITIONS, math.ceil(_NUM_GPU_TARGET_GROUPS / groups)))


def paged_decode_attention(
    q: mx.array,
    k_pool: mx.array,
    v_pool: mx.array,
    page_table: mx.array,
    table_idx: mx.array,
    lens: mx.array,
) -> mx.array:
    """
    q: [B, Hq, D]; k_pool/v_pool: [num_slots, Hkv, D]; page_table: [R, S] token slots;
    table_idx/lens: [B] row of each request and its context length (including the new token).
    Returns [B, Hq, D]. The softmax scale is 1 / sqrt(D).
    """
    bs, num_qo_heads, head_dim = q.shape
    num_kv_heads = k_pool.shape[1]
    group = num_qo_heads // num_kv_heads
    assert head_dim % 32 == 0 and num_qo_heads % num_kv_heads == 0
    nsg = max(1, min(8, 4096 // (group * head_dim)))  # keep threadgroup memory <= 16KB
    parts = num_partitions(bs, num_kv_heads)
    template = [
        ("T", q.dtype),
        ("D", head_dim),
        ("G", group),
        ("HKV", num_kv_heads),
        ("NSG", nsg),
        ("P", parts),
        ("PT_STRIDE", page_table.shape[1]),
    ]
    inputs = [q, k_pool, v_pool, page_table, table_idx, lens]
    launch = {"grid": (32 * nsg, bs * num_kv_heads, parts), "threadgroup": (32 * nsg, 1, 1)}
    if parts == 1:
        return _decode_single(
            inputs=inputs,
            template=template,
            output_shapes=[q.shape],
            output_dtypes=[q.dtype],
            **launch,
        )[0]
    part_m, part_l, part_o = _decode_split(
        inputs=inputs,
        template=template,
        output_shapes=[(bs * num_qo_heads * parts,)] * 2 + [(bs * num_qo_heads * parts, head_dim)],
        output_dtypes=[mx.float32] * 3,
        **launch,
    )
    return _combine(
        inputs=[part_m, part_l, part_o],
        template=[("T", q.dtype), ("D", head_dim), ("P", parts)],
        grid=(head_dim, bs * num_qo_heads, 1),
        threadgroup=(head_dim, 1, 1),
        output_shapes=[q.shape],
        output_dtypes=[q.dtype],
    )[0]
