const Context = @import("./context.zig").Context;
const tensor = @import("./tensor.zig");

pub const op_tensor_new = *const fn (
    ctx: *const Context,
    dtype: tensor.TensorType,
    rank: u8,
    shape: [4]usize,
    data: ?*u8,
) *tensor.Tensor;

pub const op_tensor_free = *const fn (
    ctx: *const Context,
    tensor: *tensor.Tensor,
) void;

pub const op_tensor_sum = *const fn (
    ctx: *const Context,
    tensor: *tensor.Tensor,
    dtype: tensor.TensorType,
) *tensor.Tensor;

pub const op_tensor_matmul = *const fn (
    ctx: *const Context,
    tensor: *tensor.Tensor,
    tensor_other: *tensor.Tensor,
    dtype: tensor.TensorType,
) *tensor.Tensor;

pub const op_tensor_transpose = *const fn (
    ctx: *const Context,
    tensor: *tensor.Tensor,
    dim0: u8,
    dim1: u8,
) *tensor.Tensor;

pub const op_tensor_sigmoid = *const fn (
    ctx: *const Context,
    tensor: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_softmax = *const fn (
    ctx: *const Context,
    tensor: *tensor.Tensor,
    dim: u8,
    dtype: tensor.TensorType,
) *tensor.Tensor;

pub const op_tensor_abs = *const fn (
    ctx: *const Context,
    tensor: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_acos = *const fn (
    ctx: *const Context,
    tensor: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_asin = *const fn (
    ctx: *const Context,
    tensor: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_atan = *const fn (
    ctx: *const Context,
    tensor: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_cos = *const fn (
    ctx: *const Context,
    tensor: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_sin = *const fn (
    ctx: *const Context,
    tensor: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_tan = *const fn (
    ctx: *const Context,
    tensor: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_cosh = *const fn (
    ctx: *const Context,
    tensor: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_sinh = *const fn (
    ctx: *const Context,
    tensor: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_tanh = *const fn (
    ctx: *const Context,
    tensor: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_ceil = *const fn (
    ctx: *const Context,
    tensor: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_sqrt = *const fn (
    ctx: *const Context,
    tensor: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_lgamma = *const fn (
    ctx: *const Context,
    tensor: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_reshape = *const fn (
    ctx: *const Context,
    tensor: *tensor.Tensor,
    shape: [4]usize,
) *tensor.Tensor;

pub const op_tensor_flatten = *const fn (
    ctx: *const Context,
    tensor: *tensor.Tensor,
    first_dim: u8,
    end_dim: u8,
) *tensor.Tensor;

// Both tensors must be 1D
pub const op_tensor_dot = *const fn (
    ctx: *const Context,
    tensor: *tensor.Tensor,
    other_tensor: *tensor.Tensor,
) *tensor.Tensor;

// Same as previous but if a tensor has more than one dimension finds the dot product between their last dimensions
pub const op_tensor_inner = *const fn (
    ctx: *const Context,
    tensor: *tensor.Tensor,
    other_tensor: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_foreach_abs = *const fn (
    ctx: *const Context,
    tensors: []*tensor.Tensor,
) []*tensor.Tensor;

pub const op_tensor_foreach_acos = *const fn (
    ctx: *const Context,
    tensors: []*tensor.Tensor,
) []*tensor.Tensor;

pub const op_tensor_foreach_asin = *const fn (
    ctx: *const Context,
    tensors: []*tensor.Tensor,
) []*tensor.Tensor;

pub const op_tensor_foreach_atan = *const fn (
    ctx: *const Context,
    tensors: []*tensor.Tensor,
) []*tensor.Tensor;

pub const op_tensor_foreach_cos = *const fn (
    ctx: *const Context,
    tensors: []*tensor.Tensor,
) []*tensor.Tensor;

pub const op_tensor_foreach_sin = *const fn (
    ctx: *const Context,
    tensors: []*tensor.Tensor,
) []*tensor.Tensor;

pub const op_tensor_foreach_tan = *const fn (
    ctx: *const Context,
    tensors: []*tensor.Tensor,
) []*tensor.Tensor;

pub const op_tensor_foreach_cosh = *const fn (
    ctx: *const Context,
    tensors: []*tensor.Tensor,
) []*tensor.Tensor;

pub const op_tensor_foreach_sinh = *const fn (
    ctx: *const Context,
    tensors: []*tensor.Tensor,
) []*tensor.Tensor;

pub const op_tensor_foreach_tanh = *const fn (
    ctx: *const Context,
    tensors: []*tensor.Tensor,
) []*tensor.Tensor;

pub const op_tensor_foreach_ceil = *const fn (
    ctx: *const Context,
    tensors: []*tensor.Tensor,
) []*tensor.Tensor;

pub const op_tensor_foreach_sqrt = *const fn (
    ctx: *const Context,
    tensors: []*tensor.Tensor,
) []*tensor.Tensor;

pub const op_tensor_foreach_lgamma = *const fn (
    ctx: *const Context,
    tensors: []*tensor.Tensor,
) []*tensor.Tensor;

pub const Ops = struct {
    tensor_new: op_tensor_new,
    tensor_free: op_tensor_free,
    // tensor_sum: op_tensor_sum,
    // tensor_matmul: op_tensor_matmul,
    // tensor_transpose: op_tensor_transpose,
    // tensor_sigmoid: op_tensor_sigmoid,
    // tensor_softmax: op_tensor_softmax,
    // tensor_reshape: op_tensor_reshape,
    // tensor_flatten: op_tensor_flatten,
    // tensor_dot: op_tensor_dot,
    // tensor_inner: op_tensor_inner,
    // tensor_abs: op_tensor_abs,
    // tensor_acos: op_tensor_acos,
    // tensor_asin: op_tensor_asin,
    // tensor_atan: op_tensor_atan,
    // tensor_ceil: op_tensor_ceil,
    // tensor_cos: op_tensor_cos,
    // tensor_sin: op_tensor_sin,
    // tensor_tan: op_tensor_tan,
    // tensor_cosh: op_tensor_cosh,
    // tensor_sinh: op_tensor_sinh,
    // tensor_tanh: op_tensor_tanh,
    // tensor_sqrt: op_tensor_sqrt,
    // tensor_lgamma: op_tensor_lgamma,
    // tensor_foreach_abs: op_tensor_foreach_abs,
    // tensor_foreach_acos: op_tensor_foreach_acos,
    // tensor_foreach_asin: op_tensor_foreach_asin,
    // tensor_foreach_atan: op_tensor_foreach_atan,
    // tensor_foreach_ceil: op_tensor_foreach_ceil,
    // tensor_foreach_cos: op_tensor_foreach_cos,
    // tensor_foreach_sin: op_tensor_foreach_sin,
    // tensor_foreach_tan: op_tensor_foreach_tan,
    // tensor_foreach_cosh: op_tensor_foreach_cosh,
    // tensor_foreach_sinh: op_tensor_foreach_sinh,
    // tensor_foreach_tanh: op_tensor_foreach_tanh,
    // tensor_foreach_sqrt: op_tensor_foreach_sqrt,
    // tensor_foreach_lgamma: op_tensor_foreach_lgamma,
};
