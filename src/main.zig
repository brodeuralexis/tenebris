const std = @import("std");

pub const math = @import("./math.zig");

pub fn main() anyerror!u8 {
    var v = math.Vector3(f32).UNIT;

    std.log.info("{}", .{ v });

    return 0;
}

test "tenebris" {
    std.testing.refAllDecls(@This());
}
