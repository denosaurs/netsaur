const std = @import("std");
const Backend = @import("./context.zig").Backend;
const Context = @import("./context.zig").Context;

export fn ns_context_new(backend: Backend) *Context {
    const ctx = std.heap.page_allocator.create(Context) catch unreachable;
    ctx.* = Context.init(backend);
    return ctx;
}

export fn ns_context_free(ctx: *Context) void {
    std.heap.page_allocator.destroy(ctx);
}
