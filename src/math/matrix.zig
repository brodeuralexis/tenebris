const std = @import("std");

pub fn Matrix(comptime N: usize, comptime T: type) type {
    return switch (N) {
        2 => Matrix2(T),
        3 => Matrix3(T),
        4 => Matrix4(T),
        else => @compileError(N ++ " sided matrices are not supported"),
    };
}

pub fn Matrix2(comptime T: type) type {
    return struct {
        const Self = @This();

        data: [2][2]T,

        pub usingnamespace MatrixMixin(Self, 2, T);
    };
}

pub fn Matrix3(comptime T: type) type {
    return struct {
        const Self = @This();

        data: [3][3]T,

        pub usingnamespace MatrixMixin(Self, 3, T);
    };
}

pub fn Matrix4(comptime T: type) type {
    return struct {
        const Self = @This();

        data: [4][4]T,

        pub usingnamespace MatrixMixin(Self, 4, T);
    };
}

fn MatrixMixin(comptime Self: type, comptime N_: usize, comptime T_: type) type {
    return struct {
        comptime {
            if (!std.meta.trait.isNumber(T)) {
                @compileError("matrices are only defined for numerical values");
            }
        }

        /// The size of a side of this square matrix.
        pub const N = N_;

        /// The scalar type of the elements of this matrix.
        pub const T = T_;

        /// A zero-filled matrix.
        pub const ZERO = comptime filled(0);

        /// An identity matrix.
        pub const IDENTITY = comptime diagonal(1);

        /// Creates a matrix from the given data.
        pub fn from(data: [N][N]T) callconv(.Inline) Self {
            return Self {
                .data = data,
            };
        }

        /// Creates a matrix filled with the specified value.
        pub fn filled(value: T) callconv(.Inline) Self {
            var result: Self = undefined;

            comptime var row = 0;
            inline while (row < N) : (row += 1) {
                comptime var col = 0;
                inline while (col < N) : (col += 1) {
                    result.put(row, col, value);
                }
            }

            return result;
        }

        /// Creates a diagonal matrix wit hthe specified value.
        pub fn diagonal(value: T) callconv(.Inline) Self {
            var result: Self = undefined;

            comptime var row = 0;
            inline while (row < N) : (row += 1) {
                comptime var col = 0;
                inline while (col < N) : (col += 1) {
                    result.put(row, col, if (row == col) value else 0);
                }
            }

            return result;
        }

        /// Returns the value at the given row and column.
        pub fn at(self: Self, row: usize, col: usize) callconv(.Inline) T {
            return self.data[row][col];
        }

        /// Returns a pointer to the value at the specified row and column.
        pub fn ptrAt(self: *Self, row: usize, col: usize) callconv(.Inline) *T {
            return &self.data[row][col];
        }

        /// Puts the specified value at the given row and column.
        pub fn put(self: *Self, row: usize, col: usize, value: T) callconv(.Inline) void {
            self.ptrAt(row, col).* = value;
        }

        /// Adds this matrix to another matrix.
        pub fn add(self: Self, other: Self) callconv(.Inline) Self {
            var result: Self = undefined;

            comptime var row = 0;
            inline while (row < N) : (row += 1) {
                comptime var col = 0;
                inline while (col < N) : (col += 1) {
                    result.ptrAt(row, col).* = self.at(row, col) + other.at(row, col);
                }
            }

            return result;
        }

        /// Adds another matrix to this matrix in-place.
        pub fn add_(self: *Self, other: Self) callconv(.Inline) void {
            comptime var row = 0;
            inline while (row < N) : (row += 1) {
                comptime var col = 0;
                inline while (col < N) : (col += 1) {
                    self.ptrAt(row, col).* += other.at(row, col);
                }
            }
        }

        /// Subtracts another matrix from this matrix.
        pub fn sub(self: Self, other: Self) callconv(.Inline) Self {
            var result: Self = undefined;

            comptime var row = 0;
            inline while (row < N) : (row += 1) {
                comptime var col = 0;
                inline while (col < N) : (col += 1) {
                    result.ptrAt(row, col).* = self.at(row, col) - other.at(row, col);
                }
            }

            return result;
        }

        /// Subtracts another matrix from this matrix in-place.
        pub fn sub_(self: *Self, other: Self) callconv(.Inline) void {
            comptime var row = 0;
            inline while (row < N) : (row += 1) {
                comptime var col = 0;
                inline while (col < N) : (col += 1) {
                    self.ptrAt(row, col).* -= other.at(row, col);
                }
            }
        }

        /// Multiplies this matrix by another matrix.
        pub fn mul(self: Self, other: Self) Self {
            var result: Self = undefined;

            comptime var row = 0;
            inline while (row < N) : (row += 1) {
                comptime var col = 0;
                inline while (col < N) : (col += 1) {
                    result.put(row, col, 0);

                    comptime var p = 0;
                    inline while (p < N) : (p += 1) {
                        result.ptrAt(row, col).* += self.at(row, p) * other.at(p, col);
                    }
                }
            }

            return result;
        }

        /// Multiplies this matrix by another matrix in-place.
        pub fn mul_(self: *Self, other: Self) callconv(.Inline) void {
            comptime var row = 0;
            inline while (row < N) : (row += 1) {
                // Remember to change this if the major mode changes.
                const buffer = self.data[row];

                comptime var col = 0;
                inline while (col < N) : (col += 1) {
                    self.put(row, col, 0);

                    comptime var p = 0;
                    inline while (p < N) : (p += 1) {
                        self.ptrAt(row, col).* += buffer[p] * other.at(p, col);
                    }
                }
            }
        }
    };
}

const t = std.testing;

fn expectMatrixEqual(expected: anytype, actual: @TypeOf(expected)) void {
    const M = @TypeOf(expected);

    comptime var row = 0;
    inline while (row < M.N) : (row += 1) {
        comptime var col = 0;
        inline while (col < M.N) : (col += 1) {
            t.expectEqual(expected.at(row, col), actual.at(row, col));
        }
    }
}

test "matrix zero" {
    const zero = Matrix2(f32).ZERO;
    t.expectEqual(@as(f32, 0), zero.data[0][0]);
    t.expectEqual(@as(f32, 0), zero.data[0][1]);
    t.expectEqual(@as(f32, 0), zero.data[1][0]);
    t.expectEqual(@as(f32, 0), zero.data[1][1]);
}

test "matrix identity" {
    const id = Matrix3(f32).IDENTITY;
    t.expectEqual(@as(f32, 1), id.data[0][0]);
    t.expectEqual(@as(f32, 0), id.data[0][1]);
    t.expectEqual(@as(f32, 0), id.data[1][0]);
    t.expectEqual(@as(f32, 1), id.data[1][1]);
}

test "matrix from" {
    const id = Matrix2(f32).from(.{
        .{ 1, 0 },
        .{ 0, 1 },
    });

    t.expectEqual(@as(f32, 1), id.data[0][0]);
    t.expectEqual(@as(f32, 0), id.data[0][1]);
    t.expectEqual(@as(f32, 0), id.data[1][0]);
    t.expectEqual(@as(f32, 1), id.data[1][1]);

    const m = Matrix2(f32).from(.{
        .{ 1, 2 },
        .{ 3, 4 },
    });

    t.expectEqual(@as(f32, 1), m.data[0][0]);
    t.expectEqual(@as(f32, 2), m.data[0][1]);
    t.expectEqual(@as(f32, 3), m.data[1][0]);
    t.expectEqual(@as(f32, 4), m.data[1][1]);
}

test "matrix filled" {
    const m = Matrix3(f32).filled(42);

    comptime var row = 0;
    inline while (row < 3) : (row += 1) {
        comptime var col = 0;
        inline while (col < 3) : (col += 1) {
            t.expectEqual(@as(f32, 42), m.data[row][col]);
        }
    }
}

test "matrix diagonal" {
    const d = Matrix2(f32).diagonal(13);

    comptime var row = 0;
    inline while (row < 2) : (row += 1) {
        comptime var col = 0;
        inline while (col < 2) : (col += 1) {
            if (row == col) {
                t.expectEqual(@as(f32, 13), d.data[row][col]);
            } else {
                t.expectEqual(@as(f32, 0), d.data[row][col]);
            }
        }
    }
}

test "matrix at" {
    const m = Matrix2(f32).from(.{
        .{ 1, 2 },
        .{ 3, 4 },
    });

    t.expectEqual(m.data[0][0], m.at(0, 0));
    t.expectEqual(m.data[0][1], m.at(0, 1));
    t.expectEqual(m.data[1][0], m.at(1, 0));
    t.expectEqual(m.data[1][1], m.at(1, 1));
}

test "matrix ptrAt" {
    var m = Matrix2(f32).from(.{
        .{ 1, 2 },
        .{ 3, 4 },
    });

    t.expectEqual(&m.data[0][0], m.ptrAt(0, 0));
    t.expectEqual(&m.data[0][1], m.ptrAt(0, 1));
    t.expectEqual(&m.data[1][0], m.ptrAt(1, 0));
    t.expectEqual(&m.data[1][1], m.ptrAt(1, 1));
}

test "matrix put" {
    var id = Matrix2(f32).diagonal(13);
    id.put(0, 1, 13);
    id.put(1, 0, 13);

    comptime var row = 0;
    inline while (row < 2) : (row += 1) {
        comptime var col = 0;
        inline while (col < 2) : (col += 1) {
            t.expectEqual(@as(f32, 13), id.data[row][col]);
        }
    }
}

test "matrix add" {
    const a = Matrix2(f32).from(.{
        .{ 1, 2 },
        .{ 3, 4 },
    });

    const b = Matrix2(f32).from(.{
        .{ 5, 6 },
        .{ 7, 8 },
    });

    const c = a.add(b);

    expectMatrixEqual(Matrix2(f32).from(.{
        .{ 6, 8 },
        .{ 10, 12 },
    }), c);
}

test "matrix add_" {
    var a = Matrix2(f32).from(.{
        .{ 1, 2 },
        .{ 3, 4 },
    });

    const b = Matrix2(f32).from(.{
        .{ 5, 6 },
        .{ 7, 8 },
    });

    a.add_(b);

    expectMatrixEqual(Matrix2(f32).from(.{
        .{ 6, 8 },
        .{ 10, 12 },
    }), a);
}

test "matrix sub" {
    const a = Matrix2(f32).from(.{
        .{ 1, 2 },
        .{ 3, 4 },
    });

    const b = Matrix2(f32).from(.{
        .{ 5, 6 },
        .{ 7, 8 },
    });

    const c = a.sub(b);

    expectMatrixEqual(Matrix2(f32).from(.{
        .{ -4, -4 },
        .{ -4, -4 },
    }), c);
}

test "matrix sub_" {
    var a = Matrix2(f32).from(.{
        .{ 1, 2 },
        .{ 3, 4 },
    });

    const b = Matrix2(f32).from(.{
        .{ 5, 6 },
        .{ 7, 8 },
    });

    a.sub_(b);

    expectMatrixEqual(Matrix2(f32).from(.{
        .{ -4, -4 },
        .{ -4, -4 },
    }), a);
}

test "matrix mul" {
    const a = Matrix2(f32).from(.{
        .{ 1, 2 },
        .{ 3, 4 },
    });

    const b = Matrix2(f32).from(.{
        .{ 2, 0 },
        .{ 1, 2 },
    });

    const c = a.mul(b);

    expectMatrixEqual(Matrix2(f32).from(.{
        .{ 4,  4 },
        .{ 10, 8 },
    }), c);
}

test "matrix mul_" {
    var a = Matrix2(f32).from(.{
        .{ 1, 2 },
        .{ 3, 4 },
    });

    const b = Matrix2(f32).from(.{
        .{ 2, 0 },
        .{ 1, 2 },
    });

    a.mul_(b);

    expectMatrixEqual(Matrix2(f32).from(.{
        .{ 4,  4 },
        .{ 10, 8 },
    }), a);
}
