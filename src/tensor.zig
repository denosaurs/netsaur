pub const TensorType = enum(u8) {
  int8,
  int16,
  int32,
  float16,
  float32,
};

pub const Tensor = struct {
  dtype: TensorType,
  rank: u8,
  shape: [4]usize,
  backend_data: ?*u8,
};
