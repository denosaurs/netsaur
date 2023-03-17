const std = @import("std");
const Ops = @import("./ops.zig").Ops;

pub const Backend = enum(u8) {
    cpu,
};

pub const Context = struct {
    backend: Backend,
    ops: *const Ops,
    arena: std.heap.ArenaAllocator,
    allocator: std.mem.Allocator,

    pub fn init(backend: Backend) Context {
        var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        return .{
            .backend = backend,
            .ops = switch (backend) {
                .cpu => &@import("./backends/cpu/ops.zig").ops,
            },
            .arena = arena,
            .allocator = arena.allocator(),
        };
    }
};
