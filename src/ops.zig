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

pub const Ops = struct {
  tensor_new: op_tensor_new,
  tensor_free: op_tensor_free,
};
