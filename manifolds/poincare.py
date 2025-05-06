"""Poincare ball manifold."""
import torch

from manifolds.base import Manifold
from utils.sparse_mx_utils import sparse_where_same_indices, sparse_zero_prod_dim, tanh, artanh
from torch.amp import autocast

class PoincareBall(Manifold):
    """
    PoicareBall Manifold class.

    We use the following convention: x0^2 + x1^2 + ... + xd^2 < 1 / c

    Note that 1/sqrt(c) is the Poincare ball radius.

    """

    def __init__(self, ):
        super(PoincareBall, self).__init__()
        self.name = 'PoincareBall'
        self.min_norm = 1e-15
        self.eps = {torch.float32: 4e-3, torch.float64: 1e-5}

    def sqdist(self, p1, p2, c):
        sqrt_c = c ** 0.5
        dist_c = artanh(
            sqrt_c * self.mobius_add(-p1, p2, c, dim=-1).norm(dim=-1, p=2, keepdim=False)
        )
        dist = dist_c * 2 / sqrt_c
        return dist ** 2

    def _lambda_x(self, x, c):
        x_sqnorm = torch.sum(x.data.pow(2), dim=-1, keepdim=True)
        return 2 / (1. - c * x_sqnorm).clamp_min(self.min_norm)

    def egrad2rgrad(self, p, dp, c):
        lambda_p = self._lambda_x(p, c)
        dp /= lambda_p.pow(2)
        return dp

    def proj(self, x, c):
        norm = x.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
        maxnorm = (1 - self.eps[x.dtype]) / (c ** 0.5)
        if x.is_sparse:    
            cond = norm.to_dense() > maxnorm
            print        
            norm = torch.broadcast_to(norm, x.shape)
            maxnorm = torch.broadcast_to(maxnorm, x.shape)
            cond = torch.broadcast_to(cond, x.size())
        else:
            cond = norm > maxnorm
        if x.is_sparse:
            projected = x.sparse_divide(norm, self.min_norm) * maxnorm
            projection = sparse_where_same_indices(cond, projected, x)
        else:
            projected = x / norm * maxnorm
            projection = torch.where(cond, projected, x)
        return projection

    def proj_tan(self, u, p, c):
        return u

    def proj_tan0(self, u, c):
        return u

    def expmap(self, u, p, c):
        sqrt_c = c ** 0.5
        u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        if u.is_sparse:
            second_term = tanh(sqrt_c / 2 * self._lambda_x(p, c) * u_norm) * u.sparse_divide(sqrt_c * u_norm)
        else:
            second_term = (
                    tanh(sqrt_c / 2 * self._lambda_x(p, c) * u_norm)
                    * u
                    / (sqrt_c * u_norm)
            )
        gamma_1 = self.mobius_add(p, second_term, c)
        return gamma_1

    def logmap(self, p1, p2, c):
        sub = self.mobius_add(-p1, p2, c)
        sub_norm = sub.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        lam = self._lambda_x(p1, c)
        sqrt_c = c ** 0.5
        return 2 / sqrt_c / lam * artanh(sqrt_c * sub_norm) * sub / sub_norm

    def expmap0(self, u, c):
        sqrt_c = c ** 0.5    
        u_norm = u.float().norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        u_norm = torch.broadcast_to(u_norm, (u.shape))
        if u.is_sparse:
            gamma_1 = (tanh(sqrt_c * u_norm) * u).sparse_divide(sqrt_c * u_norm)
        else:
            gamma_1 = (tanh(sqrt_c * u_norm) * u ) / (sqrt_c * u_norm)
        return gamma_1

    def logmap0(self, p, c):
        sqrt_c = c ** 0.5
        p_norm = p.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        if p.is_sparse:
            p_norm = torch.broadcast_to(p_norm, p.shape)
            scale = 1. / sqrt_c * artanh(sqrt_c * p_norm) * p.sparse_divide(p_norm)
        else:
            scale = 1. / sqrt_c * artanh(sqrt_c * p_norm) / p_norm
        return scale * p

    def mobius_add(self, x, y, c, dim=-1):
        if x.is_sparse or y.is_sparse:
            if not x.is_sparse:
                x = x.to_sparse(layout=torch.sparse_coo)
            if not y.is_sparse:
                y = y.to_sparse(layout=torch.sparse_coo)

            if x.shape > y.shape:
                y = torch.broadcast_to(y, x.shape)
                x = x.coalesce()
                
            elif y.shape > x.shape:
                x = torch.broadcast_to(x, y.shape)
                y = y.coalesce()
            
        x2 = x.pow(2).sum(dim=dim, keepdim=True)
        y2 = y.pow(2).sum(dim=dim, keepdim=True)
        xy = (x * y).sum(dim=dim, keepdim=True)
        
        if x.is_sparse or y.is_sparse:
            # x2, y2 and xy have the same shapes
            x2 = torch.broadcast_to(x2, x.shape)
            y2 = torch.broadcast_to(y2, x.shape)
            xy = torch.broadcast_to(xy, x.shape)
            # x2 = x2.coalesce()
            # c = c.unsqueeze(len(x2.shape) - 1)
            # c = torch.broadcast_to(c, x2.shape)
            
        
            
            # twos = 2 * ones
            
            # Calculate numerator components
            term1 = 2 * c * xy
            term2 = c * y2
            print("term1 is sparse", term1.is_sparse)
            print("term2 is sparse", term2.is_sparse)
            term3 = x + (term1 + term2) * x
            term4 = y - (c * x2) * y
            num = term3 + term4
            
            # Calculate denominator components
            term5 = 2 * c * xy
            term5 = term5.coalesce()
            term6 = c ** 2 * x2 * y2
            ones = torch.sparse_coo_tensor(
                indices=term5.indices(),
                values=torch.ones_like(term5.values()),
                size=term5.size(),
                device=term5.device,
                dtype=term5.dtype
                )
            denom = ones + term5 + term6
        else:
            # Original dense implementation
            num = (2 * c * xy + c * y2 + 1) * x + (1 - c * x2) * y
            denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
        if denom.is_sparse:
            return num.sparse_divide(denom, self.min_norm)        
        return num / denom.clamp_min(self.min_norm)
    
    def mobius_matvec(self, m, x, c):
        print("x shape", x.shape)
        print("m shape", m.shape)
        sqrt_c = c ** 0.5
        x_norm = x.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
        if x.is_sparse or m.is_sparse:
            if not x.is_sparse:
                x = x.to_sparse(layout=torch.sparse_coo)
            if not m.is_sparse: 
                m = m.to_sparse(layout=torch.sparse_coo)
            if x.shape[1] > m.shape[0]:
                m = torch.broadcast_to(m, (m.shape[0], x.shape[1]))
            elif m.shape[1] > x.shape[1]:
                x = torch.broadcast_to(x, (x.shape[0], m.shape[1]))  
            with autocast('cuda', enabled=False):
                mx = torch.sparse.mm(x, m.transpose(-1, -2))
        else:
            mx = x @ m.transpose(-1, -2)
        mx_norm = mx.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
        
        if x.is_sparse:
            res_c = torch.broadcast_to(tanh(mx_norm.sparse_divide(x_norm)), x_norm.shape) * artanh(sqrt_c * x_norm)
        else:
            res_c = tanh(mx_norm / x_norm * artanh(sqrt_c * x_norm))

        if mx.is_sparse:
            res_c = torch.broadcast_to(res_c, mx.shape)
            res_c = res_c * torch.broadcast_to(mx.sparse_divide(mx_norm * sqrt_c), res_c.shape)

        else:   
            res_c = res_c * mx / (mx_norm * sqrt_c)

        if mx.is_sparse:
            cond = sparse_zero_prod_dim(mx, -1, keepdim=True, dtype=torch.bool)
            cond = torch.broadcast_to(cond, res_c.size())
            res_c = res_c.coalesce()
            res_1 = torch.sparse_coo_tensor(
                res_c.indices(),
                torch.ones_like(res_c.values()),
                res_c.size(),
                device=res_c.device,
                dtype=res_c.dtype
            )
            res = sparse_where_same_indices(cond, res_c, res_1)
        else:
            cond = (mx == 0).prod(-1, keepdim=True, dtype=torch.uint8)
            res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
            res = torch.where(cond, res_0, res_c)
        return res

    def init_weights(self, w, c, irange=1e-5):
        w.data.uniform_(-irange, irange)
        return w

    def _gyration(self, u, v, w, c, dim: int = -1):
        u2 = u.pow(2).sum(dim=dim, keepdim=True)
        v2 = v.pow(2).sum(dim=dim, keepdim=True)
        uv = (u * v).sum(dim=dim, keepdim=True)
        uw = (u * w).sum(dim=dim, keepdim=True)
        vw = (v * w).sum(dim=dim, keepdim=True)
        c2 = c ** 2
        a = -c2 * uw * v2 + c * vw + 2 * c2 * uv * vw
        b = -c2 * vw * u2 - c * uw
        d = 1 + 2 * c * uv + c2 * u2 * v2
        return w + 2 * (a * u + b * v) / d.clamp_min(self.min_norm)

    def inner(self, x, c, u, v=None, keepdim=False):
        if v is None:
            v = u
        lambda_x = self._lambda_x(x, c)
        return lambda_x ** 2 * (u * v).sum(dim=-1, keepdim=keepdim)

    def ptransp(self, x, y, u, c):
        lambda_x = self._lambda_x(x, c)
        lambda_y = self._lambda_x(y, c)
        return self._gyration(y, -x, u, c) * lambda_x / lambda_y

    def ptransp_(self, x, y, u, c):
        lambda_x = self._lambda_x(x, c)
        lambda_y = self._lambda_x(y, c)
        return self._gyration(y, -x, u, c) * lambda_x / lambda_y

    def ptransp0(self, x, u, c):
        lambda_x = self._lambda_x(x, c)
        return 2 * u / lambda_x.clamp_min(self.min_norm)

    def to_hyperboloid(self, x, c):
        K = 1./ c
        sqrtK = K ** 0.5
        sqnorm = torch.norm(x, p=2, dim=1, keepdim=True) ** 2
        return sqrtK * torch.cat([K + sqnorm, 2 * sqrtK * x], dim=1) / (K - sqnorm)

