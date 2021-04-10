const std = @import("std");

pub usingnamespace std.math;

pub usingnamespace @import("./math/vector.zig");
pub usingnamespace @import("./math/matrix.zig");

test "math" {
    std.testing.refAllDecls(@This());
}
