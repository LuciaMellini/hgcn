from functools import wraps

import torch
from functools import wraps
from torch.autograd import Function
from collections import defaultdict
from utils.math_utils import tanh as _original_tanh
from utils.math_utils import artanh as _original_artanh

_original_tensor_norm = torch.Tensor.norm

class SparseNormFunction(Function):
    @staticmethod
    def forward(ctx, input_tensor, p=2, dim=None, keepdim=False):
        print("In SparseNormFunction forward")
        ctx.p = p
        ctx.dim = dim
        ctx.keepdim = keepdim
        ctx.input_shape = input_tensor.shape
        ctx.save_for_backward(input_tensor)
        if not input_tensor.is_sparse:
            return _original_tensor_norm(input_tensor, p=p, dim=dim, keepdim=keepdim)
        
        input_tensor = input_tensor.coalesce()
        values = input_tensor.values()
        indices = input_tensor.indices()

        if dim is None:
            # Global norm
            if p == 'inf':
                return torch.max(torch.abs(values))
            elif p == '-inf':
                return torch.min(torch.abs(values))
            else:
                return torch.sum(torch.abs(values) ** p).pow(1.0 / p)

        if isinstance(dim, int):
            dim = (dim,)

        # Normalize negative dimensions
        dim = tuple(d if d >= 0 else d + input_tensor.ndim for d in dim)
        reduce_dims = set(dim)
        
        # Get output shape with keepdim
        out_shape = list(input_tensor.shape)
        if not keepdim:
            # Remove dimensions that are being reduced
            for d in sorted(dim, reverse=True):
                out_shape.pop(d)
        else:
            # Set reduced dimensions to 1
            for d in dim:
                out_shape[d] = 1
        
        # Indices for grouping
        keep_dims = [i for i in range(input_tensor.ndim) if i not in reduce_dims]
        if not keep_dims:
            raise ValueError("Must retain at least one dimension")

        # Build index mapping
        group_map = defaultdict(list)
        for i in range(indices.size(1)):
            key = tuple(indices[d, i].item() for d in keep_dims)
            group_map[key].append(i)

        # Create new sparse indices and values
        new_indices = []
        new_values = []
        
        for key, idx_list in group_map.items():
            vals = values[idx_list]
            if p == 'inf':
                norm_val = torch.max(torch.abs(vals))
            elif p == '-inf':
                norm_val = torch.min(torch.abs(vals))
            else:
                norm_val = torch.sum(torch.abs(vals) ** p).pow(1.0 / p)
            
            # Create the indices for this result value
            idx = list(key)
            # Insert 0's for reduced dimensions if keeping dims
            if keepdim:
                for d in sorted(dim):
                    idx.insert(d, 0)
            new_indices.append(idx)
            new_values.append(norm_val)
        
        if not new_indices:
            return torch.sparse_coo_tensor(torch.zeros(len(out_shape), 0), torch.zeros(0), 
                                           size=torch.Size(out_shape), device=values.device)
        
        sparse_indices = torch.tensor(new_indices, device=indices.device).t()
        sparse_values = torch.tensor(new_values, device=values.device)
        
        return torch.sparse_coo_tensor(sparse_indices, sparse_values, torch.Size(out_shape))
    
    @staticmethod
    def backward(ctx, grad_output):
        print("In SparseNormFunction backward")
        input_tensor, = ctx.saved_tensors
        p = ctx.p
        dim = ctx.dim
        keepdim = ctx.keepdim
        
        if not input_tensor.is_sparse:
            print("input tensor is dense")
            # Use standard backward for dense tensors
            if p == 'inf' or p == '-inf':
                # Handle inf norms with min/max operations
                if p == 'inf':
                    # For inf norm, gradient flows only through elements with max absolute value
                    abs_input = torch.abs(input_tensor)
                    max_val = torch.max(abs_input, dim=dim, keepdim=True)[0]
                    mask = abs_input == max_val
                    # Keep sign of the original input
                    grad_input = torch.zeros_like(input_tensor)
                    grad_input[mask] = torch.sign(input_tensor[mask])
                    print("p inf")
                    return grad_input * grad_output, None, None, None
                elif p == '-inf':
                    # For -inf norm, gradient flows only through elements with min absolute value
                    abs_input = torch.abs(input_tensor)
                    min_val = torch.min(abs_input, dim=dim, keepdim=True)[0] 
                    mask = abs_input == min_val
                    # Keep sign of the original input
                    grad_input = torch.zeros_like(input_tensor)
                    grad_input[mask] = torch.sign(input_tensor[mask])
                    print("p -inf")
                    return grad_input * grad_output, None, None, None
                
            norm = _original_tensor_norm(input_tensor, p=p, dim=dim, keepdim=True)
            if not keepdim and dim is not None:
                norm = norm.unsqueeze(dim) if isinstance(dim, int) else norm
            norm = norm.clamp_min(1e-8)
            grad_input = input_tensor * torch.abs(input_tensor)**(p-2) / norm**(p-1)
            if not keepdim and dim is not None:
                if isinstance(dim, int):
                    grad_output = grad_output.unsqueeze(dim)
                else:
                    for d in sorted(dim):
                        grad_output = grad_output.unsqueeze(d)
            return grad_input * grad_output, None, None, None
        
        # Handle sparse tensors
        if p == 'inf' or p == '-inf':
            # Handle inf norms for sparse tensors
            input_tensor = input_tensor.coalesce()
            indices = input_tensor.indices()
            values = input_tensor.values()
            
            if p == 'inf':
            # For inf norm, gradient flows only through elements with max absolute value
                abs_values = torch.abs(values)
                max_val = torch.max(abs_values)
                mask = abs_values == max_val
                grad_values = torch.zeros_like(values)
                grad_values[mask] = torch.sign(values[mask])
                return torch.sparse_coo_tensor(indices, grad_values * grad_output.item(), input_tensor.size()), None, None, None
            elif p == '-inf':
                # For -inf norm, gradient flows only through elements with min absolute value
                abs_values = torch.abs(values)
                min_val = torch.min(abs_values)
                mask = abs_values == min_val
                grad_values = torch.zeros_like(values)
                grad_values[mask] = torch.sign(values[mask])
                return torch.sparse_coo_tensor(indices, grad_values * grad_output.item(), input_tensor.size()), None, None, None
        
        input_tensor = input_tensor.coalesce()
        indices = input_tensor.indices()
        values = input_tensor.values()
        
        if dim is None:
            # Global norm
            norm = torch.sum(torch.abs(values) ** p).pow(1.0 / p)
            grad_values = grad_output * torch.abs(values)**(p-2) * values / norm**(p-1)
            return torch.sparse_coo_tensor(indices, grad_values, input_tensor.size()), None, None, None
        
        # Create mapping from output to input
        if isinstance(dim, int):
            dim = (dim,)
        
        # Normalize negative dimensions
        dim = tuple(d if d >= 0 else d + input_tensor.ndim for d in dim)
        reduce_dims = set(dim)
        
        # Indices for grouping
        keep_dims = [i for i in range(input_tensor.ndim) if i not in reduce_dims]
        
        # Build group map: which input indices correspond to which output position
        group_map = defaultdict(list)
        for i in range(indices.size(1)):
            key = tuple(indices[d, i].item() for d in keep_dims)
            group_map[key].append(i)
        
        # For each input value, compute its gradient contribution
        grad_values = torch.zeros_like(values)
        grad_output = grad_output.coalesce() if grad_output.is_sparse else grad_output
        
        for key, idx_list in group_map.items():
            vals = values[idx_list]
            norm = torch.sum(torch.abs(vals) ** p).pow(1.0 / p)
            
            # Find corresponding gradient in grad_output
            lookup_idx = list(key)
            
            if keepdim:
                # If keepdim, we need to insert 0s for the reduced dimensions
                for d in sorted(dim):
                    lookup_idx.insert(d, 0)
            
            # Get gradient from grad_output
            if grad_output.is_sparse:
                # Look up in sparse grad_output
                grad_val = 0.0
                grad_indices = grad_output.indices()
                for j in range(grad_indices.size(1)):
                    match = True
                    for dim_idx, val in enumerate(lookup_idx):
                        if grad_indices[dim_idx, j].item() != val:
                            match = False
                            break
                    if match:
                        grad_val = grad_output.values()[j].item()
                        break
            else:
                # Look up in dense grad_output
                grad_val = grad_output[tuple(lookup_idx)].item()
            
            # Calculate gradients for all values in this group
            for i, idx in enumerate(idx_list):
                val = vals[i]
                grad_values[idx] = grad_val * torch.abs(val)**(p-2) * val / norm**(p-1)
        
        return torch.sparse_coo_tensor(indices, grad_values, input_tensor.size()), None, None, None

def _sparse_norm(self, p=2, dim=None, keepdim=False):
    """
    A norm function that handles both dense and sparse tensors correctly.
    """
    return SparseNormFunction.apply(self, p, dim, keepdim)

torch.Tensor.norm = _sparse_norm


class SparseDivideFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, divisor, min_norm=1e-8):
        print("In SparseDivideFunction forward")
        is_input_sparse = input_tensor.is_sparse
        ctx.is_input_sparse = is_input_sparse
        
        if is_input_sparse:
            input_tensor = input_tensor.coalesce()
            indices = input_tensor.indices()
            values = input_tensor.values()
            shape = input_tensor.shape
        else:
            # Handle dense input tensor
            shape = input_tensor.shape
            values = input_tensor

        if isinstance(divisor, (int, float)):
            ctx.save_for_backward(input_tensor)
            ctx.divisor = divisor
            ctx.divisor_type = "scalar"
            ctx.min_norm = min_norm
            
            scaled_values = values / divisor
            
            if is_input_sparse:
                return torch.sparse_coo_tensor(indices, scaled_values, size=shape, device=values.device)
            else:
                return scaled_values
            
        elif not divisor.is_sparse and (divisor.dim() == 1 or divisor.dim() == 2):
            ctx.save_for_backward(input_tensor, divisor)
            ctx.divisor_type = "dense"
            ctx.min_norm = min_norm
            
            if is_input_sparse:
                row_idx = indices[0]
                divisor_values = divisor[row_idx].squeeze().clamp_min(min_norm)
                print("line 283")
                print("values shape", values.shape)
                print("divisor shape", divisor_values.shape)
                scaled_values = values / divisor_values
                ctx.row_idx = row_idx
                return torch.sparse_coo_tensor(indices, scaled_values, size=shape, device=values.device)
            else:
                # For dense input, broadcast division
                divisor = divisor.clamp_min(min_norm)
                print("line 290")
                print("input tensor shape", input_tensor.shape)
                print("input tensor sparse", input_tensor.is_sparse)
                print("divisor shape", divisor.shape)
                print("divisor is sparse", divisor.is_sparse)
                return input_tensor / divisor
            
        elif divisor.is_sparse:
            divisor = divisor.coalesce()
            divisor_indices = divisor.indices()
            divisor_values = divisor.values()
            
            ctx.save_for_backward(input_tensor, divisor)
            ctx.divisor_type = "sparse"
            ctx.min_norm = min_norm
            
            if is_input_sparse:
                divisor_map = torch.zeros_like(values)
                for idx in range(divisor_indices.shape[1]):
                    divisor_map[indices[0] == divisor_indices[0][idx]] = divisor_values[idx]
                
                divisor_map = divisor_map.clamp_min(min_norm)
                scaled_values = values.sparse_divide(divisor_map)
                print("line 315")
                return torch.sparse_coo_tensor(indices, scaled_values, size=shape, device=values.device)
            else:
                # Convert sparse divisor to dense for dense input
                print("to dense")
                print("divisor shape", divisor.shape)
                dense_divisor = divisor.to_dense().clamp_min(min_norm)
                return input_tensor / dense_divisor
        else:
            raise ValueError(f"Unsupported divisor type: {type(divisor)}")
    
    @staticmethod
    def backward(ctx, grad_output):
        print("In SparseDivideFunction backward")
        min_norm = ctx.min_norm
        is_input_sparse = ctx.is_input_sparse
        
        if ctx.divisor_type == "scalar":
            input_tensor, = ctx.saved_tensors
            divisor = ctx.divisor
            
            # Gradient w.r.t. input is grad_output / divisor
            grad_input = grad_output / divisor if is_input_sparse else grad_output / divisor
            
            # Gradient w.r.t. divisor is -input * grad_output / divisor^2
            if is_input_sparse:
                input_tensor = input_tensor.coalesce()
                grad_divisor = -(input_tensor.values() * grad_output.coalesce().values() / (divisor * divisor)).sum()
            else:
                grad_divisor = -(input_tensor * grad_output / (divisor * divisor)).sum()
            
            return grad_input, grad_divisor, None
            
        elif ctx.divisor_type == "dense":
            if len(ctx.saved_tensors) == 2:
                input_tensor, divisor = ctx.saved_tensors
                
                if is_input_sparse:
                    input_tensor = input_tensor.coalesce()
                    indices = input_tensor.indices()
                    values = input_tensor.values()
                    
                    if grad_output.is_sparse:
                        grad_output = grad_output.coalesce()
                        grad_values = grad_output.values()
                    else:
                        # Get sparse gradient values at the sparse input locations
                        grad_values = grad_output[tuple(idx for idx in indices)]
                    
                    # Gradient w.r.t. input
                    row_idx = indices[0] if hasattr(ctx, 'row_idx') else indices[0]
                    divisor_values = divisor[row_idx].squeeze().clamp_min(min_norm)
                    print("line 356")
                    grad_input_values = grad_values / divisor_values
                    grad_input = torch.sparse_coo_tensor(indices, grad_input_values, input_tensor.size())
                    
                    # Gradient w.r.t. divisor
                    grad_divisor = torch.zeros_like(divisor)
                    # For each value, accumulate gradient contributions
                    for i in range(indices.size(1)):
                        row = indices[0, i].item()
                        div_val = divisor[row].item()
                        grad_divisor[row] -= (values[i] * grad_values[i]) / (div_val * div_val)
                else:
                    # Dense input case
                    divisor = divisor.clamp_min(min_norm)
                    print("line 370")
                    grad_input = grad_output / divisor
                    
                    # For dense input and dense divisor
                    grad_divisor = -(input_tensor * grad_output / (divisor * divisor))
                    # Sum across all dimensions except those matching the divisor
                    reduce_dims = tuple(i for i in range(grad_divisor.ndim) if i >= divisor.ndim)
                    if reduce_dims:
                        grad_divisor = grad_divisor.sum(dim=reduce_dims)
                
                return grad_input, grad_divisor, None
            
        elif ctx.divisor_type == "sparse":
            input_tensor, divisor = ctx.saved_tensors
            divisor = divisor.coalesce()
            
            if not is_input_sparse:
                input_tensor = input_tensor.to_sparse(layout=torch.sparse_coo)
            input_tensor = input_tensor.coalesce()
            indices = input_tensor.indices()
            values = input_tensor.values()
            
            if grad_output.is_sparse:
                grad_output = grad_output.coalesce()
                grad_values = grad_output.values()
            else:
                # Extract values at sparse locations
                grad_values = grad_output[tuple(idx for idx in indices)]
            
            # Create divisor map for the sparse locations
            divisor_map = torch.zeros_like(values)
            divisor_indices = divisor.indices()
            divisor_values = divisor.values()
            
            for idx in range(divisor_indices.shape[1]):
                mask = indices[0] == divisor_indices[0][idx]
                divisor_map[mask] = divisor_values[idx]
            print("line 407")
            divisor_map = divisor_map.clamp_min(min_norm)
            
            # Gradient w.r.t. input
            
            grad_input_values = grad_values.sparse_divide(divisor_map)
            print("line 424")
            grad_input = torch.sparse_coo_tensor(indices, grad_input_values, input_tensor.size())
            
            # Gradient w.r.t. divisor
            # Line 408
            grad_divisor_values = torch.zeros_like(divisor_values)
            for idx in range(divisor_indices.shape[1]):
                row = divisor_indices[0][idx].item()
                mask = indices[0] == row
                if mask.any():
                    div_val = divisor_values[idx]
                    contrib = -(values[mask] * grad_values[mask]) / (div_val * div_val)
 
                    grad_divisor_values[idx] = contrib.sum()
            print("line 417")
            print("grad divisor values shape", grad_divisor_values.shape)
            print("divisor indices", divisor_indices.shape)
            grad_divisor = torch.sparse_coo_tensor(divisor_indices, grad_divisor_values, divisor.size())
                        
            return grad_input, grad_divisor, None
        
        # Default case - should not reach here
        return None, None, None

def _sparse_divide(self, divisor, min_norm=1e-8):
    """
    Division operation for sparse or dense tensors with backpropagation support.
    
    Args:
        divisor: A scalar, dense tensor, or sparse tensor to divide by
        min_norm: Minimum value to clamp the divisor (to avoid division by zero)
        
    Returns:
        A tensor representing the result of the division (sparse if input is sparse)
    """
    return SparseDivideFunction.apply(self, divisor, min_norm)

torch.Tensor.sparse_divide = _sparse_divide

def _memory_efficient_where(condition, x, y):
    """
    Memory-efficient implementation of torch.where that avoids
    creating unnecessary intermediate tensors.
    
    Args:
        condition (torch.Tensor): Boolean tensor
        x (torch.Tensor): Values to use where condition is True
        y (torch.Tensor): Values to use where condition is False
    
    Returns:
        torch.Tensor: Result equivalent to torch.where(condition, x, y)
    """
    result = y.clone()
    mask = condition.bool().squeeze()
    result[mask] = x[mask]

    return result
class SparseWhereSameIndicesFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cond, x_sparse, y_sparse):
        print("In SparseWhereSameIndicesFunction forward")

        # Ensure sparse tensors are coalesced
        x_sparse = x_sparse.coalesce()
        y_sparse = y_sparse.coalesce()
        shape = torch.Size(x_sparse.shape)

        indices = x_sparse.indices()
        x_vals = x_sparse.values()
        y_vals = y_sparse.values()
        

        idx_tuples = [indices[i] for i in range(indices.shape[0])]
        cond_vals = cond[idx_tuples]
        
        # Apply the condition to select values
        result_vals = _memory_efficient_where(cond_vals, x_vals, y_vals)
        
        # Save for backward
        ctx.save_for_backward(cond_vals, indices)
        ctx.x_shape = x_sparse.shape
        ctx.y_shape = y_sparse.shape
        
        return torch.sparse_coo_tensor(indices, result_vals, shape)

    @staticmethod
    def backward(ctx, grad_output):
        print("In SparseWhereSameIndicesFunction backward")
        cond_vals, indices = ctx.saved_tensors
        grad_output = grad_output.coalesce()
        grad_vals = grad_output.values()
        
        # Where condition is True, gradient flows to x
        # Where condition is False, gradient flows to y
        grad_x = _memory_efficient_where(cond_vals, grad_vals, torch.zeros_like(grad_vals))
        grad_y = _memory_efficient_where(cond_vals, torch.zeros_like(grad_vals), grad_vals)
        
        grad_x_sparse = torch.sparse_coo_tensor(indices, grad_x, ctx.x_shape)
        grad_y_sparse = torch.sparse_coo_tensor(indices, grad_y, ctx.y_shape)
        
        # No gradient for condition
        return None, grad_x_sparse, grad_y_sparse

def sparse_where_same_indices(cond, x_sparse, y_sparse):
    """
    Mimic torch.where for sparse tensors x and y that share the same indices,
    with support for backpropagation.
    
    Args:
        cond (torch.Tensor): Boolean tensor (broadcast to match x and y's values).
        x (torch.Tensor): First tensor (sparse or dense).
        y (torch.Tensor): Second tensor (sparse or dense).
    
    Returns:
        torch.Tensor: Result of the conditional where operation, sparse if any input is sparse.
    """
    return SparseWhereSameIndicesFunction.apply(cond, x_sparse, y_sparse)


_original_clamp_min = torch.Tensor.clamp_min

class SparseClampMinFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, min_value):
        print("In SparseClampMinFunction forward")
        ctx.min_value = min_value
        ctx.save_for_backward(input_tensor)
        
        if not input_tensor.is_sparse:
            return _original_clamp_min(input_tensor, min=min_value)
        input_tensor = input_tensor.coalesce()
        values = _original_clamp_min(input_tensor.values(), min=min_value)
        return torch.sparse_coo_tensor(input_tensor.indices(), values, input_tensor.size())
    
    @staticmethod
    def backward(ctx, grad_output):
        print("In SparseClampMinFunction backward")
        input_tensor, = ctx.saved_tensors
        min_value = ctx.min_value
        
        if not input_tensor.is_sparse:
            grad_input = grad_output.clone()
            mask = (input_tensor < min_value).squeeze()
            grad_input[mask] = 0
            return grad_input, None
        
        input_tensor = input_tensor.coalesce()
        grad_output = grad_output.coalesce()
        
        # Only propagate gradients where input > min_value
        mask = input_tensor.values() >= min_value
        print("to dense")
        print("grad output shape", grad_output.shape)
        grad_values = torch.where(mask, grad_output.to_dense(), torch.zeros_like(grad_output.to_dense()))
        
        grad_input = grad_values.to_sparse(layout=torch.sparse_coo)
        
        return grad_input, None

@wraps(_original_clamp_min)
def clamp_min_wrapper(x, min_value: float):
    return SparseClampMinFunction.apply(x, min_value)

torch.Tensor.clamp_min = clamp_min_wrapper


class SparseZeroProdDimFunction(Function):
    @staticmethod
    def forward(ctx, input_tensor, dim=-1, keepdim=True, dtype=torch.uint8):
        """Forward pass for sparse_zero_prod_dim with context saving for backward"""
        print("In SparseZeroProdDimFunction forward")
        input_tensor = input_tensor.coalesce()
        ctx.save_for_backward(input_tensor)
        ctx.dim = dim
        ctx.keepdim = keepdim
        ctx.dtype = dtype
        ctx.input_shape = input_tensor.shape

        # Original implementation logic
        ndim = len(input_tensor.shape)
        if dim < 0:
            dim = ndim + dim
        
        ctx.actual_dim = dim  
        
        if keepdim:
            output_shape = list(input_tensor.shape)
            output_shape[dim] = 1
        else:
            output_shape = list(input_tensor.shape)
            output_shape.pop(dim)
        
        result = torch.ones(output_shape, dtype=dtype, device=input_tensor.device)
        
        indices = input_tensor._indices()
        
        if indices.size(1) == 0:  
            return result

        groups = []
        for i in range(ndim):
            if i != dim:
                groups.append(indices[i])

        if not groups:
            return result

        grouped_indices = torch.stack(groups, dim=0)
        unique_groups = torch.unique(grouped_indices, dim=1)
        
        for i in range(unique_groups.size(1)):
            idx = []
            group_idx = 0
            for d in range(ndim):
                if d != dim:
                    idx.append(unique_groups[group_idx, i].item())
                    group_idx += 1
                elif keepdim:
                    idx.append(0)
            result[tuple(idx)] = 0
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass for sparse_zero_prod_dim"""
        print("In SparseZeroProdDimFunction backward")
        input_tensor, = ctx.saved_tensors
        dim = ctx.actual_dim
        keepdim = ctx.keepdim
        
        # For zero product, the gradient is zero everywhere because:
        # If any element is zero, the product is zero and the gradient is zero
        # If no elements are zero, the output is one and changes in input don't affect it
        
        # Create a sparse tensor with the same indices but zero values
        input_tensor = input_tensor.coalesce()
        indices = input_tensor._indices()
        values = torch.zeros_like(input_tensor._values())
        
        # Gradient with respect to input is a sparse tensor with same shape as input but all zeros
        grad_input = torch.sparse_coo_tensor(
            indices, 
            values, 
            input_tensor.size(),
            device=input_tensor.device
        )
        
        # No gradients for the other parameters
        return grad_input, None, None, None
    
def sparse_zero_prod_dim(self, dim=-1, keepdim=True, dtype=torch.uint8):
    """
    Generalized version of (self == 0).prod(dim, keepdim=keepdim, dtype=dtype) for sparse tensors
    with backpropagation support.
    
    Args:
        self: A sparse tensor
        dim: Dimension along which to compute the product (default: -1)
        keepdim: Whether to keep the reduced dimension (default: True)
        dtype: Data type of the output tensor (default: torch.uint8)
        
    Returns:
        Dense tensor with values 0 or 1
    """
    return SparseZeroProdDimFunction.apply(self, dim, keepdim, dtype)


class SparseTanhFunction(Function):
    @staticmethod
    def forward(ctx, input_tensor, clamp=15):
        print("In SparseTanhFunction forward")
        if input_tensor.is_sparse:
            input_tensor = input_tensor.coalesce()
            ctx.is_sparse = True
            ctx.shape = input_tensor.size()
            ctx.indices = input_tensor.indices()
            values = _original_tanh(input_tensor.values(), clamp=clamp)
            ctx.save_for_backward(values)
            return torch.sparse_coo_tensor(ctx.indices, values, ctx.shape)
        else:
            ctx.is_sparse = False
            output = _original_tanh(input_tensor, clamp=clamp)
            ctx.save_for_backward(output)
            return output
    
    @staticmethod
    def backward(ctx, grad_output):
        print("In SparseTanhFunction backward")
        if ctx.is_sparse:
            values, = ctx.saved_tensors
            grad_values = grad_output.coalesce().values() * (1 - values.pow(2))
            grad_input = torch.sparse_coo_tensor(ctx.indices, grad_values, ctx.shape)
        else:
            output, = ctx.saved_tensors
            grad_input = grad_output * (1 - output.pow(2))
        
        return grad_input, None

@wraps(_original_tanh)
def tanh_wrapper(x, clamp=15):
    return SparseTanhFunction.apply(x, clamp)

tanh = tanh_wrapper


@wraps(_original_artanh)
def artanh_wrapper(x, clamp=15):
    if not x.is_sparse:
        return _original_artanh(x)
    x = x.coalesce()
    return torch.sparse_coo_tensor(x.indices(), _original_artanh(x.values()), x.size())

artanh = artanh_wrapper

_original_broadcast_to = torch.broadcast_to

@wraps(_original_broadcast_to)
def _sparse_broadcast_to(self, shape):
    if not self.is_sparse:
        return _original_broadcast_to(self, shape)
    self = self.coalesce()
    return torch.sparse_coo_tensor(
        self.indices(),
        self.values(),
        shape
    )
    
torch.broadcast_to = _sparse_broadcast_to