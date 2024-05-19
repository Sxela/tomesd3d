import torch
from typing import Tuple, Callable


def do_nothing(x: torch.Tensor, mode:str=None):
    return x


def mps_gather_workaround(input, dim, index):
    if input.shape[-1] == 1:
        return torch.gather(
            input.unsqueeze(-1),
            dim - 1 if dim < 0 else dim,
            index.unsqueeze(-1)
        ).squeeze(-1)
    else:
        return torch.gather(input, dim, index)


def bipartite_soft_matching_random2d(metric: torch.Tensor,
                                     w: int, h: int, sx: int, sy: int, r: int,
                                     no_rand: bool = False,
                                     generator: torch.Generator = None) -> Tuple[Callable, Callable]:
    """
    Partitions the tokens into src and dst and merges r tokens from src to dst.
    Dst tokens are partitioned by choosing one randomy in each (sx, sy) region.

    Args:
     - metric [B, N, C]: metric to use for similarity
     - w: image width in tokens
     - h: image height in tokens
     - sx: stride in the x dimension for dst, must divide w
     - sy: stride in the y dimension for dst, must divide h
     - r: number of tokens to remove (by merging)
     - no_rand: if true, disable randomness (use top left corner only)
     - rand_seed: if no_rand is false, and if not None, sets random seed.
    """
    B, N, _ = metric.shape
    # print('B N w h sx sy r', B, N, w, h, sx,sy, r)

    if r <= 0:
        return do_nothing, do_nothing

    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather
    
    with torch.no_grad():
        hsy, wsx = h // sy, w // sx
        # print('hsy, wsx', hsy, wsx)

        # For each sy by sx kernel, randomly assign one token to be dst and the rest src
        if no_rand:
            rand_idx = torch.zeros(hsy, wsx, 1, device=metric.device, dtype=torch.int64)
        else:
            rand_idx = torch.randint(sy*sx, size=(hsy, wsx, 1), device=generator.device, generator=generator).to(metric.device)

        # print('rand_idx', rand_idx.shape)
        
        # The image might not divide sx and sy, so we need to work on a view of the top left if the idx buffer instead
        idx_buffer_view = torch.zeros(hsy, wsx, sy*sx, device=metric.device, dtype=torch.int64)
        # print('idx_buffer_view', idx_buffer_view.shape)
        idx_buffer_view.scatter_(dim=2, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype))
        # print('idx_buffer_view', idx_buffer_view.shape)
        idx_buffer_view = idx_buffer_view.view(hsy, wsx, sy, sx).transpose(1, 2).reshape(hsy * sy, wsx * sx)
        # print('idx_buffer_view', idx_buffer_view.shape)

        # Image is not divisible by sx or sy so we need to move it into a new buffer
        if (hsy * sy) < h or (wsx * sx) < w:
            idx_buffer = torch.zeros(h, w, device=metric.device, dtype=torch.int64)
            idx_buffer[:(hsy * sy), :(wsx * sx)] = idx_buffer_view
        else:
            idx_buffer = idx_buffer_view
        # print('idx_buffer', idx_buffer.shape)

        # We set dst tokens to be -1 and src to be 0, so an argsort gives us dst|src indices
        rand_idx = idx_buffer.reshape(1, -1, 1).argsort(dim=1)
        # print('rand_idx', rand_idx.shape)

        # We're finished with these
        del idx_buffer, idx_buffer_view

        # rand_idx is currently dst|src, so split them
        num_dst = hsy * wsx
        # print('num_dst', num_dst)
        a_idx = rand_idx[:, num_dst:, :] # src
        b_idx = rand_idx[:, :num_dst, :] # dst
        # print('a_idx', a_idx)
        # print('b_idx', b_idx)

        def split(x):
            C = x.shape[-1]
            src = gather(x, dim=1, index=a_idx.expand(B, N - num_dst, C))
            dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
            return src, dst

        # Cosine similarity between A and B
        metric = metric / metric.norm(dim=-1, keepdim=True)
        # print('metric', metric.shape)
        a, b = split(metric)
        # print('a', a.shape)
        # print('b', b.shape)
        scores = a @ b.transpose(-1, -2)
        # print('scores', scores.shape)

        # Can't reduce more than the # tokens in src
        r = min(a.shape[1], r)
        # print('r', r)

        # Find the most similar greedily
        node_max, node_idx = scores.max(dim=-1)
        # print('node_max', node_max)
        # print('node_idx', node_idx)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        # print('edge_idx', edge_idx)

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        # print('unm_idx', unm_idx)
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        # print('src_idx', src_idx)
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)
        # print('dst_idx', dst_idx)

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        n, t1, c = src.shape
        # print('n', n)
        # print('t1', t1)
        # print('c', c)
        
        unm = gather(src, dim=-2, index=unm_idx.expand(n, t1 - r, c))
        # print('unm', unm.shape)
        src = gather(src, dim=-2, index=src_idx.expand(n, r, c))
        # print('src', src.shape)
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
        # print('dst', dst.shape)

        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        _, _, c = unm.shape

        src = gather(dst, dim=-2, index=dst_idx.expand(B, r, c))

        # Combine back to the original shape
        out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(B, num_dst, c), src=dst)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx).expand(B, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=src_idx).expand(B, r, c), src=src)

        return out
    
    assert False

    return merge, unmerge



#v0.1 24.1.24
def bipartite_soft_matching_random2d(metric: torch.Tensor,
                                     w: int, h: int, sx: int, sy: int, r: int,
                                     d: int = 0, sz: int = 0, 
                                     no_rand: bool = False,
                                     generator: torch.Generator = None):
    """
    Partitions the tokens into src and dst and merges r tokens from src to dst.
    Dst tokens are partitioned by choosing one randomy in each (sx, sy) region.

    Args:
     - metric [B, N, C]: metric to use for similarity
     - w: image width in tokens
     - h: image height in tokens
     - sx: stride in the x dimension for dst, must divide w
     - sy: stride in the y dimension for dst, must divide h
     - r: number of tokens to remove (by merging)
     - no_rand: if true, disable randomness (use top left corner only)
     - rand_seed: if no_rand is false, and if not None, sets random seed.
    """
    
    # print('metric', metric.shape)
    B, N, _ = metric.shape
    # print('B N', B, N)
    if d == 0:
      d = B
    if sz == 0:
      sz = d
    r = r*B
    metric = metric.view((1,B*N,-1))
    Bnb, Nnb, _ = metric.shape
    # print('metric', metric.shape)

    if r <= 0:
        return do_nothing, do_nothing

    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather
    
    with torch.no_grad():
        hsy, wsx, dsz = h // sy, w // sx, d // sz
        # print('hsy, wsx', hsy, wsx)

        # For each sy by sx kernel, randomly assign one token to be dst and the rest src
        if no_rand:
            rand_idx = torch.zeros(hsy, wsx, dsz, 1, device=metric.device, dtype=torch.int64)
        else:
            rand_idx = torch.randint(sy*sx*sz, size=(hsy, wsx, dsz, 1), device=generator.device, generator=generator).to(metric.device)

        # print('rand_idx', rand_idx.shape)
        
        # The image might not divide sx and sy, so we need to work on a view of the top left if the idx buffer instead
        idx_buffer_view = torch.zeros(hsy, wsx, dsz, sy*sx*sz, device=metric.device, dtype=torch.int64)
        # print('idx_buffer_view', idx_buffer_view.shape)
        idx_buffer_view.scatter_(dim=3, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype))
        # print('idx_buffer_view', idx_buffer_view.shape)
        idx_buffer_view = idx_buffer_view.view(hsy, wsx, dsz, sy, sx, sz).transpose(1, 2).reshape(hsy * sy, wsx * sx, dsz * sz)
        # print('idx_buffer_view', idx_buffer_view.shape)

        # Image is not divisible by sx or sy so we need to move it into a new buffer
        if (hsy * sy) < h or (wsx * sx) < w:
            idx_buffer = torch.zeros(h, w, device=metric.device, dtype=torch.int64)
            idx_buffer[:(hsy * sy), :(wsx * sx)] = idx_buffer_view
        else:
            idx_buffer = idx_buffer_view
        # print('idx_buffer', idx_buffer.shape)

        # We set dst tokens to be -1 and src to be 0, so an argsort gives us dst|src indices
        rand_idx = idx_buffer.reshape(1, -1, 1).argsort(dim=1)
        # print('rand_idx', rand_idx.shape)

        # We're finished with these
        del idx_buffer, idx_buffer_view

        # rand_idx is currently dst|src, so split them
        num_dst = hsy * wsx
        # print('num_dst', num_dst)
        a_idx = rand_idx[:, num_dst:, :] # src
        b_idx = rand_idx[:, :num_dst, :] # dst
        # print('a_idx', a_idx.shape)
        # print('b_idx', b_idx.shape)

        def split(x):
            # print('x.shape', x.shape)
            Bx, Nx, C = x.shape
            # C = x.shape[-1]
            src = gather(x, dim=1, index=a_idx.expand(Bx, Nx - num_dst, C))
            dst = gather(x, dim=1, index=b_idx.expand(Bx, num_dst, C))
            return src, dst

        # Cosine similarity between A and B
        metric = metric / metric.norm(dim=-1, keepdim=True)
        # print('metric', metric.shape)
        a, b = split(metric)
        # print('a', a.shape)
        # print('b', b.shape)
        scores = a @ b.transpose(-1, -2)
        # print('scores', scores.shape)

        # Can't reduce more than the # tokens in src
        r = min(a.shape[1], r)
        # print('r', r)

        # Find the most similar greedily
        node_max, node_idx = scores.max(dim=-1)
        # print('node_max', node_max.shape)
        # print('node_idx', node_idx.shape)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        # print('edge_idx', edge_idx.shape)

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        # print('unm_idx', unm_idx.shape)
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        # print('src_idx', src_idx.shape)
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)
        # print('dst_idx', dst_idx.shape)

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        x = x.view((1,B*N,-1))
        src, dst = split(x)
        n, t1, c = src.shape
        # print('n', n)
        # print('t1', t1)
        # print('c', c)
        
        unm = gather(src, dim=-2, index=unm_idx.expand(n, t1 - r, c))
        # print('unm', unm.shape)
        src = gather(src, dim=-2, index=src_idx.expand(n, r, c))
        # print('src', src.shape)
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
        # print('dst', dst.shape)

        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        _, _, c = unm.shape

        src = gather(dst, dim=-2, index=dst_idx.expand(Bnb, r, c))

        # Combine back to the original shape
        out = torch.zeros(Bnb, Nnb, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(Bnb, num_dst, c), src=dst)
        out.scatter_(dim=-2, index=gather(a_idx.expand(Bnb, a_idx.shape[1], 1), dim=1, index=unm_idx).expand(Bnb, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=gather(a_idx.expand(Bnb, a_idx.shape[1], 1), dim=1, index=src_idx).expand(Bnb, r, c), src=src)
        out = out.view((B,N,-1))
        return out
    
    # assert False

    return merge, unmerge


#v0.2 25.1.24 added depth everywhere
def bipartite_soft_matching_random2d(metric: torch.Tensor,
                                     w: int, h: int, sx: int, sy: int, r: int,
                                     d: int = 0, sz: int = 0,
                                     no_rand: bool = False,
                                     generator: torch.Generator = None):
    """
    Partitions the tokens into src and dst and merges r tokens from src to dst.
    Dst tokens are partitioned by choosing one randomy in each (sx, sy) region.

    Args:
     - metric [B, N, C]: metric to use for similarity
     - w: image width in tokens
     - h: image height in tokens
     - sx: stride in the x dimension for dst, must divide w
     - sy: stride in the y dimension for dst, must divide h
     - r: number of tokens to remove (by merging)
     - no_rand: if true, disable randomness (use top left corner only)
     - rand_seed: if no_rand is false, and if not None, sets random seed.
    """

    # print('metric', metric.shape)
    B, N, _ = metric.shape
    # print('B N', B, N)
    if d == 0:
      d = B
    if sz == 0:
      sz = d
    r = r*B
    metric = metric.view((1,B*N,-1))
    Bnb, Nnb, _ = metric.shape
    # print('metric', metric.shape)

    if r <= 0:
        return do_nothing, do_nothing

    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather

    with torch.no_grad():
        hsy, wsx, dsz = h // sy, w // sx, d // sz
        # print('hsy, wsx', hsy, wsx)

        # For each sy by sx kernel, randomly assign one token to be dst and the rest src
        if no_rand:
            rand_idx = torch.zeros(hsy, wsx, dsz, 1, device=metric.device, dtype=torch.int64)
        else:
            rand_idx = torch.randint(sy*sx*sz, size=(hsy, wsx, dsz, 1), device=generator.device, generator=generator).to(metric.device)

        # print('rand_idx', rand_idx.shape)

        # The image might not divide sx and sy, so we need to work on a view of the top left if the idx buffer instead
        idx_buffer_view = torch.zeros(hsy, wsx, dsz, sy*sx*sz, device=metric.device, dtype=torch.int64)
        # print('idx_buffer_view', idx_buffer_view.shape)
        idx_buffer_view.scatter_(dim=3, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype))
        # print('idx_buffer_view', idx_buffer_view.shape)
        idx_buffer_view = idx_buffer_view.view(hsy, wsx, dsz, sy, sx, sz).transpose(1, 2).reshape(hsy * sy, wsx * sx, dsz * sz)
        # print('idx_buffer_view', idx_buffer_view.shape)

        # Image is not divisible by sx or sy so we need to move it into a new buffer
        if (hsy * sy) < h or (wsx * sx) < w or (dsz * sz) < d:
            idx_buffer = torch.zeros(h, w, d, device=metric.device, dtype=torch.int64)
            idx_buffer[:(hsy * sy), :(wsx * sx), :(dsz*sz)] = idx_buffer_view
        else:
            idx_buffer = idx_buffer_view
        # print('idx_buffer', idx_buffer.shape)

        # We set dst tokens to be -1 and src to be 0, so an argsort gives us dst|src indices
        rand_idx = idx_buffer.reshape(1, -1, 1).argsort(dim=1)
        # print('rand_idx', rand_idx.shape)

        # We're finished with these
        del idx_buffer, idx_buffer_view

        # rand_idx is currently dst|src, so split them
        num_dst = hsy * wsx * dsz
        # print('num_dst', num_dst)
        a_idx = rand_idx[:, num_dst:, :] # src
        b_idx = rand_idx[:, :num_dst, :] # dst
        # print('a_idx', a_idx.shape)
        # print('b_idx', b_idx.shape)

        def split(x):
            # print('x.shape', x.shape)
            Bx, Nx, C = x.shape
            # C = x.shape[-1]
            src = gather(x, dim=1, index=a_idx.expand(Bx, Nx - num_dst, C))
            dst = gather(x, dim=1, index=b_idx.expand(Bx, num_dst, C))
            return src, dst

        # Cosine similarity between A and B
        metric = metric / metric.norm(dim=-1, keepdim=True)
        # print('metric', metric.shape)
        a, b = split(metric)
        # print('a', a.shape)
        # print('b', b.shape)
        scores = a @ b.transpose(-1, -2)
        # print('scores', scores.shape)

        # Can't reduce more than the # tokens in src
        r = min(a.shape[1], r)
        # print('r', r)

        # Find the most similar greedily
        node_max, node_idx = scores.max(dim=-1)
        # print('node_max', node_max.shape)
        # print('node_idx', node_idx.shape)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        # print('edge_idx', edge_idx.shape)

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        # print('unm_idx', unm_idx.shape)
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        # print('src_idx', src_idx.shape)
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)
        # print('dst_idx', dst_idx.shape)

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        x = x.view((1,B*N,-1))
        src, dst = split(x)
        n, t1, c = src.shape
        # print('n', n)
        # print('t1', t1)
        # print('c', c)

        unm = gather(src, dim=-2, index=unm_idx.expand(n, t1 - r, c))
        # print('unm', unm.shape)
        src = gather(src, dim=-2, index=src_idx.expand(n, r, c))
        # print('src', src.shape)
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
        # print('dst', dst.shape)

        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        _, _, c = unm.shape

        src = gather(dst, dim=-2, index=dst_idx.expand(Bnb, r, c))

        # Combine back to the original shape
        out = torch.zeros(Bnb, Nnb, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(Bnb, num_dst, c), src=dst)
        out.scatter_(dim=-2, index=gather(a_idx.expand(Bnb, a_idx.shape[1], 1), dim=1, index=unm_idx).expand(Bnb, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=gather(a_idx.expand(Bnb, a_idx.shape[1], 1), dim=1, index=src_idx).expand(Bnb, r, c), src=src)
        out = out.view((B,N,-1))
        return out

    # assert False

    return merge, unmerge