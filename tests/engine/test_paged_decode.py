import pytest

mx = pytest.importorskip("mlx.core")

from macsgl.engine.kernel import num_partitions, paged_decode_attention  # noqa: E402


def _reference(q, k_pool, v_pool, page_table, table_idx, lens):
    max_len = int(lens.max().item())
    idx = page_table[table_idx][:, :max_len]
    k = k_pool[idx].transpose(0, 2, 1, 3)
    v = v_pool[idx].transpose(0, 2, 1, 3)
    mask = (mx.arange(max_len)[None] < lens[:, None])[:, None, None, :]
    scale = q.shape[-1] ** -0.5
    o = mx.fast.scaled_dot_product_attention(q[:, :, None, :], k, v, scale=scale, mask=mask)
    return o[:, :, 0, :]


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("group", [1, 2, 4, 8])
@pytest.mark.parametrize("bs", [1, 3, 64])
def test_paged_decode_matches_reference(dtype, head_dim, group, bs):
    mx.random.seed(bs * 1000 + group * 10 + head_dim)
    num_kv_heads, num_slots, max_seq_len, rows = 4, 8192, 1500, 80
    k_pool = mx.random.normal((num_slots, num_kv_heads, head_dim)).astype(dtype)
    v_pool = mx.random.normal((num_slots, num_kv_heads, head_dim)).astype(dtype)
    q = mx.random.normal((bs, num_kv_heads * group, head_dim)).astype(dtype)
    # every row maps positions to scattered slots, like a fragmented paged cache
    page_table = mx.stack([mx.random.permutation(num_slots)[:max_seq_len] for _ in range(rows)])
    page_table = page_table.astype(mx.int32)
    table_idx = mx.random.permutation(rows)[:bs].astype(mx.int32)
    lens = mx.random.randint(1, max_seq_len + 1, (bs,)).astype(mx.int32)
    lens[0] = 1  # a single-token context leaves most partitions empty

    out = paged_decode_attention(q, k_pool, v_pool, page_table, table_idx, lens)
    ref = _reference(q, k_pool, v_pool, page_table, table_idx, lens)
    atol = 2e-2 if dtype == mx.bfloat16 else 1e-4
    assert out.shape == q.shape and out.dtype == dtype
    assert mx.allclose(out.astype(mx.float32), ref.astype(mx.float32), atol=atol).item()


def test_partitions_only_depend_on_batch_size():
    assert num_partitions(1, 8) > 1
    assert num_partitions(256, 8) == 1
