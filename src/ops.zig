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
    input: *tensor.Tensor,
) void;

pub const op_tensor_sum = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
    dtype: tensor.TensorType,
) *tensor.Tensor;

pub const op_tensor_add = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
    other: *tensor.Tensor,
    alpha: u8,
) *tensor.Tensor;

pub const op_tensor_sub = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
    other: *tensor.Tensor,
    alpha: u8,
) *tensor.Tensor;

pub const op_tensor_sinc = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_matmul = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
    other: *tensor.Tensor,
    dtype: tensor.TensorType,
) *tensor.Tensor;

pub const op_tensor_transpose = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
    dim0: u8,
    dim1: u8,
) *tensor.Tensor;

pub const op_tensor_sigmoid = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_softmax = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
    dim: u8,
    dtype: tensor.TensorType,
) *tensor.Tensor;

pub const op_tensor_round = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
    num_dec: u8,
) *tensor.Tensor;

pub const op_tensor_abs = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_neg = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_angle = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_sign = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_acos = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_asin = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_atan = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_cos = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_sin = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_tan = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_cosh = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_sinh = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_tanh = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_acosh = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_asinh = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_atanh = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_ceil = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_clamp = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
    min: f32,
    max: f32,
) *tensor.Tensor;

pub const op_tensor_conj_phys = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_copysign = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
    other: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_sqrt = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_rsqrt = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_lgamma = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_log = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_log10 = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_log1p = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_log2 = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_reshape = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
    shape: [4]usize,
) *tensor.Tensor;

pub const op_tensor_flatten = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
    first_dim: u8,
    end_dim: u8,
) *tensor.Tensor;

pub const op_tensor_mul = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
    other: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_div = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
    other: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_digamma = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_pow = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
    exponent: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_pow_float = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
    exponent: f32,
) *tensor.Tensor;

pub const op_tensor_exp = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_exp2 = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_expm1 = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_floor = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_floor_divide = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
    other: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_trunc = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_lerp = *const fn (
    ctx: *const Context,
    start: *tensor.Tensor,
    end: *tensor.Tensor,
    weight: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_lerp_float = *const fn (
    ctx: *const Context,
    start: *tensor.Tensor,
    end: *tensor.Tensor,
    weight: f32,
) *tensor.Tensor;

// Both tensors must be 1D
pub const op_tensor_dot = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
    other: *tensor.Tensor,
) *tensor.Tensor;

// Same as previous but if a tensor has more than one dimension finds the dot product between their last dimensions
pub const op_tensor_inner = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
    other: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_igamma = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
    other: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_igammac = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
    other: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_max = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_min = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_erf = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_erfc = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_erfcx = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_erfinv = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_convert_deg_rad = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_logical_and = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
    other: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_logical_not = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_logical_or = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
    other: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_logical_xor = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
    other: *tensor.Tensor,
) *tensor.Tensor;

pub const op_tensor_logit = *const fn (
    ctx: *const Context,
    input: *tensor.Tensor,
    epsilon: f32,
) *tensor.Tensor;

pub const Ops = struct {
    tensor_new: op_tensor_new,
    tensor_free: op_tensor_free,
    // tensor_sum: op_tensor_sum,
    // tensor_add: op_tensor_add,
    // tensor_sub: op_tensor_sub,
    // tensor_sinc: op_tensor_sinc,
    // tensor_matmul: op_tensor_matmul,
    // tensor_transpose: op_tensor_transpose,
    // tensor_sigmoid: op_tensor_sigmoid,
    // tensor_softmax: op_tensor_softmax,
    // tensor_round: op_tensor_round,
    // tensor_abs: op_tensor_abs,
    // tensor_neg: op_tensor_neg,
    // tensor_angle: op_tensor_angle,
    // tensor_sign: op_tensor_sign,
    // tensor_acos: op_tensor_acos,
    // tensor_asin: op_tensor_asin,
    // tensor_atan: op_tensor_atan,
    // tensor_cos: op_tensor_cos,
    // tensor_sin: op_tensor_sin,
    // tensor_tan: op_tensor_tan,
    // tensor_cosh: op_tensor_cosh,
    // tensor_sinh: op_tensor_sinh,
    // tensor_tanh: op_tensor_tanh,
    // tensor_acosh: op_tensor_acosh,
    // tensor_asinh: op_tensor_asinh,
    // tensor_atanh: op_tensor_atanh,
    // tensor_ceil: op_tensor_ceil,
    // tensor_clamp: op_tensor_clamp,
    // tensor_conj_phys: op_tensor_conj_phys,
    // tensor_copysign: op_tensor_copysign,
    // tensor_sqrt: op_tensor_sqrt,
    // tensor_rsqrt: op_tensor_rsqrt,
    // tensor_lgamma: op_tensor_lgamma,
    // tensor_log: op_tensor_log,
    // tensor_log10: op_tensor_log10,
    // tensor_log1p: op_tensor_log1p,
    // tensor_log2: op_tensor_log2,
    // tensor_reshape: op_tensor_reshape,
    // tensor_flatten: op_tensor_flatten,
    // tensor_mul: op_tensor_mul,
    // tensor_div: op_tensor_div,
    // tensor_digamma: op_tensor_digamma,
    // tensor_pow: op_tensor_pow,
    // tensor_pow_float: op_tensor_pow_float,
    // tensor_exp: op_tensor_exp,
    // tensor_exp2: op_tensor_exp2,
    // tensor_expm1: op_tensor_expm1,
    // tensor_floor: op_tensor_floor,
    // tensor_floor_divide: op_tensor_floor_divide,
    // tensor_trunc: op_tensor_trunc,
    // tensor_lerp: op_tensor_lerp,
    // tensor_lerp_float: op_tensor_lerp_float,
    // tensor_dot: op_tensor_dot,
    // tensor_inner: op_tensor_inner,
    // tensor_igamma: op_tensor_igamma,
    // tensor_igammac: op_tensor_igammac,
    // tensor_max: op_tensor_max,
    // tensor_min: op_tensor_min,
    // tensor_erf: op_tensor_erf,
    // tensor_erfc: op_tensor_erfc,
    // tensor_erfinv: op_tensor_erfinv,
    // tensor_convert_deg_rad: op_tensor_convert_deg_rad,
    // tensor_logical_and: op_tensor_logical_and,
    // tensor_logical_not: op_tensor_logical_not,
    // tensor_logical_or: op_tensor_logical_or,
    // tensor_logical_xor: op_tensor_logical_xor,
    // tensor_logit: op_tensor_logit,
};
