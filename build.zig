const std = @import("std");

const Builder = std.build.Builder;

pub fn build(b: *Builder) void {
    const target = b.standardTargetOptions(.{});
    const mode = b.standardReleaseOptions();

    const exe = b.addExecutable("tenebris", "src/main.zig");
    exe.setTarget(target);
    exe.setBuildMode(mode);
    exe.install();

    const run_cmd = exe.run();
    if (b.args) |args| run_cmd.addArgs(args);

    const run_step = b.step("run", "Runs the application");
    run_step.dependOn(&run_cmd.step);

    const test_exe = b.addTest("src/main.zig");
    test_exe.setBuildMode(mode);

    const test_step = b.step("test", "Tests the application");
    test_step.dependOn(&test_exe.step);
}
