const Ops = @import("../../ops.zig").Ops;
const Context = @import("../../context.zig").Context;
const tensor = @import("../../tensor.zig");

pub fn tensor_new(
  ctx: *const Context,
  dtype: tensor.TensorType,
  rank: u8,
  shape: [4]usize,
  data: ?*u8,
) *tensor.Tensor {
  const t = ctx.allocator.create(tensor.Tensor) catch unreachable;
  t.* = .{
    .dtype = dtype,
    .rank = rank,
    .shape = shape,
    .backend_data = data,
  };
  return t;
}

pub fn tensor_free(ctx: *const Context, t: *tensor.Tensor) void {
  ctx.allocator.destroy(t);
}

pub const ops = Ops {
  .tensor_new = tensor_new,
  .tensor_free = tensor_free,
};
