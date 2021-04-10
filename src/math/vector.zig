const std = @import("std");
const trait = std.meta.trait;

/// A dimensional vector of a given size and scalar type.
pub fn Vector(comptime N: usize, comptime T: type) type {
    return switch (N) {
        2 => Vector2(T),
        3 => Vector3(T),
        4 => Vector4(T),
        else => @compileError(N ++ " dimensional vectors are not supported"),
    };
}

/// A 2-dimensional vector of a given scalar type.
pub fn Vector2(comptime T: type) type {
    return struct {
        const Self = @This();

        /// The x component of this 2D vector.
        x: T = 0,
        /// The y component of this 2D vector.
        y: T = 0,

        pub usingnamespace VectorMixin(Self, 2, T);

        /// A unit vector on the x axis.
        pub const UNIT_X = Self {
            .x = 1,
        };

        /// A unit vector on the y axis.
        pub const UNIT_Y = Self {
            .y = 1,
        };

        /// Calculates the cross product between this vector and another vector.
        pub fn cross(self: Self, other: Self) callconv(.Inline) T {
            return self.x * other.y - self.y * other.x;
        }

        /// Extends this vector with another component.
        pub fn extend(self: Self, z: T) callconv(.Inline) Vector3(T) {
            return .{
                .x = self.x,
                .y = self.y,
                .z = z,
            };
        }

        /// Returns the angle between the 2 vectors.
        pub fn angle(self: Self, other: Self) callconv(.Inline) T {
            return std.math.atan2(T, self.cross(other), self.dot(other));
        }
    };
}

/// A 3 dimensional vector of a given scalar type.
pub fn Vector3(comptime T: type) type {
    return struct {
        const Self = @This();

        /// The x component of this 3D vector.
        x: T = 0,
        /// The y component of this 3D vector.
        y: T = 0,
        /// The z component of this 3D vector.
        z: T = 0,

        pub usingnamespace VectorMixin(Self, 3, T);

        /// A unit vector on the x axis.
        pub const UNIT_X = Self {
            .x = 1,
        };

        /// A unit vector on the y axis.
        pub const UNIT_Y = Self {
            .y = 1,
        };

        /// A unit vector on the z axis.
        pub const UNIT_Z = Self {
            .z = 1,
        };

        /// Calculates the cross product between this vector and another vector.
        pub fn cross(self: Self, other: Self) callconv(.Inline) Self {
            return .{
                .x = self.y * other.z - self.z * other.y,
                .y = self.z * other.x - self.x * other.z,
                .z = self.x * other.y - self.y * other.x,
            };
        }

        /// Calculates the cross product between this vector and another vector
        /// in-place.
        pub fn cross_(self: *Self, other: Self) callconv(.Inline) void {
            const x = self.x;
            const y = self.y;
            const z = self.z;

            self.x = y * other.z - z * other.y;
            self.y = z * other.x - x * other.z;
            self.z = x * other.y - y * other.x;
        }

        /// Extends this vector with another component.
        pub fn extend(self: Self, w: T) callconv(.Inline) Vector4(T) {
            return .{
                .x = self.x,
                .y = self.y,
                .z = self.z,
                .w = w,
            };
        }

        /// Truncates the last component of this vector.
        pub fn truncate(self: Self) callconv(.Inline) Vector2(T) {
            return .{
                .x = self.x,
                .y = self.y,
            };
        }

        /// Returns the angle between the 2 vectors.
        pub fn angle(self: Self, other: Self) callconv(.Inline) T {
            return std.math.atan2(T, self.cross(other).magnitude(), self.dot(other));
        }
    };
}

/// A 4 dimensional vector of a given scalar type.
pub fn Vector4(comptime T: type) type {
    return struct {
        const Self = @This();

        /// The x component of this 4D vector.
        x: T,
        /// The y component of this 4D vector.
        y: T,
        /// The z component of this 4D vector.
        z: T,
        /// The w component of this 4D vector.
        w: T,

        pub usingnamespace VectorMixin(Self, 4, T);

        /// A unit vector on the x axis.
        pub const UNIT_X = Self {
            .x = 1,
        };

        /// A unit vector on the y axis.
        pub const UNIT_Y = Self {
            .y = 1,
        };

        /// A unit vector on the z axis.
        pub const UNIT_Z = Self {
            .z = 1,
        };

        /// A unit vector on the w axis.
        pub const UNIT_W = Self {
            .w = 1,
        };

        /// Truncates the last component of this vector.
        pub fn truncate(self: Self) callconv(.Inline) Vector3(T) {
            return .{
                .x = self.x,
                .y = self.y,
                .z = self.z,
            };
        }
    };
}

fn VectorMixin(comptime Self: type, comptime N_: usize, comptime T_: type) type {
    return struct {
        comptime {
            if (!trait.isNumber(T)) {
                @compileError("vectors are only defined for numerical values");
            }
        }

        const fields = std.meta.fields(Self);

        /// The size of this vector.
        pub const N = N_;

        /// The scalar type of this vector.
        pub const T = T_;

        /// A zero-filled matrix.
        pub const ZERO = comptime filled(0);

        /// A one-filled matrix.
        pub const ONE = comptime filled(1);

        /// A unit-vector with all components having the same value.
        pub const UNIT = comptime filled(std.math.sqrt(1.0 / @as(T, N)));

        /// Creates a vector filled with the specified value.
        pub fn filled(value: T) callconv(.Inline) Self {
            var result: Self = undefined;

            inline for (fields) |field| {
                @field(result, field.name) = value;
            }

            return result;
        }

        /// Casts all components of this vector to another type.
        pub fn cast(self: Self, comptime U: type) callconv(.Inline) Vector(N, U) {
            comptime {
                if (!trait.isNumber(U)) {
                    @compileError(@typeName(U) ++ " must be a number type");
                }
            }

            if (T == U) {
                return self;
            }

            var result: Vector(N, U) = undefined;

            inline for (fields) |field| {
                if (comptime trait.isFloat(U) and trait.isIntegral(T)) {
                    @field(result, field.name) = @intToFloat(U, @field(self, field.name));
                } else if (comptime trait.isIntegral(U) and trait.isFloat(T)) {
                    @field(result, field.name) = @floatToInt(U, @field(self, field.name));
                } else if (comptime trait.isFloat(U) and trait.isFloat(T)) {
                    @field(result, field.name) = @floatCast(U, @field(self, field.name));
                } else {
                    @field(result, field.name) = @intCast(U, @field(self, field.name));
                }
            }

            return result;
        }

        /// Negates all the components of this vector.
        pub fn neg(self: Self) callconv(.Inline) Self {
            var result: Self = undefined;

            inline for (fields) |field| {
                @field(result, field.name) = -@field(self, field.name);
            }

            return result;
        }

        /// Negates all the components of this vector in-place.
        pub fn neg_(self: *Self) callconv(.Inline) void {
            inline for (fields) |field| {
                @field(self, field.name) = -@field(self, field.name);
            }
        }

        /// Adds a vector to another vector.
        pub fn add(self: Self, other: Self) callconv(.Inline) Self {
            var result: Self = undefined;

            inline for (fields) |field| {
                @field(result, field.name) = @field(self, field.name) + @field(other, field.name);
            }

            return result;
        }

        /// Adds a vector to another vector in-place.
        pub fn add_(self: *Self, other: Self) callconv(.Inline) void {
            inline for (fields) |field| {
                @field(self, field.name) += @field(other, field.name);
            }
        }

        /// Adds a vector to a scalar.
        pub fn sadd(self: Self, scalar: T) callconv(.Inline) Self {
            var result: Self = undefined;

            inline for (fields) |field| {
                @field(result, field.name) = @field(self, field.name) + scalar;
            }

            return result;
        }

        /// Adds a vector to a scalar in-place.
        pub fn sadd_(self: *Self, scalar: T) callconv(.Inline) void {
            inline for (fields) |field| {
                @field(self, field.name) += scalar;
            }
        }

        /// Subtracts another vector from this vector.
        pub fn sub(self: Self, other: Self) callconv(.Inline) Self {
            var result: Self = undefined;

            inline for (fields) |field| {
                @field(result, field.name) = @field(self, field.name) - @field(other, field.name);
            }

            return result;
        }

        /// Subtracts another from this vector in-place.
        pub fn sub_(self: *Self, other: Self) callconv(.Inline) void {
            inline for (fields) |field| {
                @field(self, field.name) -= @field(other, field.name);
            }
        }

        /// Subtracts scalar from this vector.
        pub fn ssub(self: Self, scalar: T) callconv(.Inline) Self {
            var result: Self = undefined;

            inline for (fields) |field| {
                @field(result, field.name) = @field(self, field.name) - scalar;
            }

            return result;
        }

        /// Subtracts a scalar from this vector in-place.
        pub fn ssub_(self: *Self, scalar: T) callconv(.Inline) void {
            inline for (fields) |field| {
                @field(self, field.name) -= scalar;
            }
        }

        /// Scales a vector by another vector.
        pub fn scale(self: Self, other: Self) callconv(.Inline) Self {
            var result: Self = undefined;

            inline for (fields) |field| {
                @field(result, field.name) = @field(self, field.name) * @field(other, field.name);
            }

            return result;
        }

        /// Scales this vector by another vector in-place.
        pub fn scale_(self: *Self, other: Self) callconv(.Inline) void {
            inline for (fields) |field| {
                @field(self, field.name) *= @field(other, field.name);
            }
        }

        /// Scales a vector by a scalar.
        pub fn sscale(self: Self, scalar: T) callconv(.Inline) Self {
            var result: Self = undefined;

            inline for (fields) |field| {
                @field(result, field.name) = @field(self, field.name) * scalar;
            }

            return result;
        }

        /// Scales this vector by a scalar in-place.
        pub fn sscale_(self: *Self, scalar: T) callconv(.Inline) void {
            inline for (fields) |field| {
                @field(self, field.name) *= scalar;
            }
        }

        /// Unscales this vector by another vector.
        pub fn unscale(self: Self, other: Self) callconv(.Inline) Self {
            var result: Self = undefined;

            inline for (fields) |field| {
                @field(result, field.name) = @field(self, field.name) / @field(other, field.name);
            }

            return result;
        }

        /// Unscales this vector by another vector in-place.
        pub fn unscale_(self: *Self, other: Self) callconv(.Inline) void {
            inline for (fields) |field| {
                @field(self, field.name) /= @field(other, field.name);
            }
        }

        /// Unscales this vector by a scalar.
        pub fn sunscale(self: Self, scalar: T) callconv(.Inline) Self {
            var result: Self = undefined;

            inline for (fields) |field| {
                @field(result, field.name) = @field(self, field.name) / scalar;
            }

            return result;
        }

        /// Unscales this vector by a scalar in-place.
        pub fn sunscale_(self: *Self, scalar: T) callconv(.Inline) void {
            inline for (fields) |field| {
                @field(self, field.name) /= scalar;
            }
        }

        /// Indicates if the 2 vectors are equal.
        pub fn eq(self: Self, other: Self) callconv(.Inline) bool {
            inline for (fields) |field| {
                if (@field(self, field.name) != @field(other, field.name)) {
                    return false;
                }
            }

            return true;
        }

        /// Returns the dot product between this vector and another vector.
        pub fn dot(self: Self, other: Self) callconv(.Inline) T {
            var result: T = 0;

            inline for (fields) |field| {
                result += @field(self, field.name) * @field(other, field.name);
            }

            return result;
        }

        /// Returns the square of the vector's magnitude.
        ///
        /// This is faster for comparisons then using the `magnitude` function.
        pub fn magnitude2(self: Self) callconv(.Inline) T {
            return self.dot(self);
        }

        /// Returns this vector's magnitude.
        ///
        /// This is slower but more accurate than using the `magnitude2`
        /// function.
        pub fn magnitude(self: Self) callconv(.Inline) T {
            return std.math.sqrt(self.magnitude2());
        }

        /// Returns the projection of this vector onto another vector.
        pub fn projectOn(self: Self, other: Self) callconv(.Inline) Self {
            return other.sscale(self.dot(other) / other.magnitude2());
        }

        /// Projects this vector onto another vector in-place.
        pub fn projectOn_(self: *Self, other: Self) callconv(.Inline) void {
            const a = self.dot(other);
            self.* = other.sscale(a / other.magnitude2());
        }

        /// Normalizes the length of this vector to 1.
        pub fn normalize(self: Self) callconv(.Inline) Self {
            return self.normalizeTo(1);
        }

        /// Normalizes the length of this vector to 1 in-place.
        pub fn normalize_(self: *Self) callconv(.Inline) void {
            self.normalizeTo_(1);
        }

        /// Normalizes the length of this vector to the specified magnitude.
        pub fn normalizeTo(self: Self, mag: T) callconv(.Inline) Self {
            return self.sscale(mag / self.magnitude());
        }

        /// Normalizes the length of this vector to the specified magnitude
        /// in-place.
        pub fn normalizeTo_(self: *Self, mag: T) callconv(.Inline) void {
            self.sscale_(mag / self.magnitude());
        }
    };
}

const t = std.testing;

fn expectVectorEquals(expected: anytype, actual: @TypeOf(expected)) void {
    const M = @TypeOf(expected);
    inline for (std.meta.fields(M)) |field| {
        const expected_field = @field(expected, field.name);
        const actual_field = @field(actual, field.name);

        if (expected_field != actual_field) {
            std.debug.panic("`expected.{s}` is not equal to `actual.{s}`.  Got `{} == {}`", .{
                field.name,
                field.name,
                expected_field,
                actual_field,
            });
        }
    }
}

fn expectVectorApproxEq(expected: anytype, actual: @TypeOf(expected), comptime tolerance: comptime_float) void {
    const M = @TypeOf(expected);
    inline for (std.meta.fields(M)) |field| {
        const expected_field = @field(expected, field.name);
        const actual_field = @field(actual, field.name);

        if (std.math.fabs(expected_field - actual_field) > tolerance) {
            std.debug.panic("`expected.{s}` is not equal to `actual.{s}` with a tolerance of `{}`.  Got `{} == {}`", .{
                field.name,
                field.name,
                tolerance,
                expected_field,
                actual_field,
            });
        }
    }
}

test "vector zero" {
    expectVectorEquals(Vector3(f32){
        .x = 0,
        .y = 0,
        .z = 0,
    }, Vector3(f32).ZERO);
}

test "vector one" {
    expectVectorEquals(Vector3(f32){
        .x = 1,
        .y = 1,
        .z = 1,
    }, Vector3(f32).ONE);
}

test "vector unit" {
    expectVectorEquals(Vector3(f32){
        .x = std.math.sqrt(1.0 / 3.0),
        .y = std.math.sqrt(1.0 / 3.0),
        .z = std.math.sqrt(1.0 / 3.0),
    }, Vector3(f32).UNIT);
}

test "vector filled" {
    expectVectorEquals(
        Vector3(f32){
            .x = 42,
            .y = 42,
            .z = 42,
        },
        Vector3(f32).filled(42),
    );
}

test "vector cast" {
    const a = Vector2(i32){
        .x = 42,
        .y = 13,
    };

    const b = a.cast(f64);

    expectVectorEquals(
        Vector2(f64){
            .x = 42,
            .y = 13,
        },
        b,
    );

    const c = b.cast(f32);

    expectVectorEquals(
        Vector2(f32){
            .x = 42,
            .y = 13,
        },
        c,
    );

    const d = c.cast(u8);

    expectVectorEquals(
        Vector2(u8){
            .x = 42,
            .y = 13,
        },
        d,
    );

    const e = d.cast(isize);

    expectVectorEquals(
        Vector2(isize){
            .x = 42,
            .y = 13,
        },
        e,
    );
}

test "vector neg" {
    const a = Vector3(f32){
        .x = 1,
        .y = 2,
        .z = 3,
    };

    expectVectorEquals(
        Vector3(f32){
            .x = -1,
            .y = -2,
            .z = -3,
        },
        a.neg(),
    );
}

test "vector neg_" {
    var a = Vector3(f32){
        .x = 1,
        .y = 2,
        .z = 3,
    };

    a.neg_();

    expectVectorEquals(
        Vector3(f32){
            .x = -1,
            .y = -2,
            .z = -3,
        },
        a,
    );
}

test "vector add" {
    const a = Vector3(f32){
        .x = 1,
        .y = 2,
        .z = 3,
    };

    const b = Vector3(f32){
        .x = 4,
        .y = 5,
        .z = 6,
    };

    expectVectorEquals(
        Vector3(f32){
            .x = 5,
            .y = 7,
            .z = 9,
        },
        a.add(b),
    );
}

test "vector add_" {
    var a = Vector3(f32){
        .x = 1,
        .y = 2,
        .z = 3,
    };

    const b = Vector3(f32){
        .x = 4,
        .y = 5,
        .z = 6,
    };

    a.add_(b);

    expectVectorEquals(
        Vector3(f32){
            .x = 5,
            .y = 7,
            .z = 9,
        },
        a,
    );
}

test "vector sadd" {
    const a = Vector3(f32){
        .x = 1,
        .y = 2,
        .z = 3,
    };

    expectVectorEquals(
        Vector3(f32){
            .x = 4,
            .y = 5,
            .z = 6,
        },
        a.sadd(3),
    );
}

test "vector sadd_" {
    var a = Vector3(f32){
        .x = 1,
        .y = 2,
        .z = 3,
    };

    a.sadd_(3);

    expectVectorEquals(
        Vector3(f32){
            .x = 4,
            .y = 5,
            .z = 6,
        },
        a,
    );
}

test "vector sub" {
    const a = Vector3(f32){
        .x = 1,
        .y = 2,
        .z = 3,
    };

    const b = Vector3(f32){
        .x = 4,
        .y = 5,
        .z = 6,
    };

    expectVectorEquals(
        Vector3(f32){
            .x = -3,
            .y = -3,
            .z = -3,
        },
        a.sub(b),
    );
}

test "vector sub_" {
    var a = Vector3(f32){
        .x = 1,
        .y = 2,
        .z = 3,
    };

    const b = Vector3(f32){
        .x = 4,
        .y = 5,
        .z = 6,
    };

    a.sub_(b);

    expectVectorEquals(
        Vector3(f32){
            .x = -3,
            .y = -3,
            .z = -3,
        },
        a,
    );
}

test "vector ssub" {
    const a = Vector3(f32){
        .x = 1,
        .y = 2,
        .z = 3,
    };

    expectVectorEquals(
        Vector3(f32){
            .x = -2,
            .y = -1,
            .z = 0,
        },
        a.ssub(3),
    );
}

test "vector sub_" {
    var a = Vector3(f32){
        .x = 1,
        .y = 2,
        .z = 3,
    };

    a.ssub_(3);

    expectVectorEquals(
        Vector3(f32){
            .x = -2,
            .y = -1,
            .z = 0,
        },
        a,
    );
}

test "vector scale" {
    const a = Vector3(f32){
        .x = 1,
        .y = 2,
        .z = 3,
    };

    const b = Vector3(f32){
        .x = 4,
        .y = 5,
        .z = 6,
    };

    expectVectorEquals(
        Vector3(f32){
            .x = 4,
            .y = 10,
            .z = 18,
        },
        a.scale(b),
    );
}

test "vector scale_" {
    var a = Vector3(f32){
        .x = 1,
        .y = 2,
        .z = 3,
    };

    const b = Vector3(f32){
        .x = 4,
        .y = 5,
        .z = 6,
    };

    a.scale_(b);

    expectVectorEquals(
        Vector3(f32){
            .x = 4,
            .y = 10,
            .z = 18,
        },
        a,
    );
}

test "vector sscale" {
    const a = Vector3(f32){
        .x = 1,
        .y = 2,
        .z = 3,
    };

    expectVectorEquals(
        Vector3(f32){
            .x = 10,
            .y = 20,
            .z = 30,
        },
        a.sscale(10),
    );
}

test "vector sscale_" {
    var a = Vector3(f32){
        .x = 1,
        .y = 2,
        .z = 3,
    };

    a.sscale_(10);

    expectVectorEquals(
        Vector3(f32){
            .x = 10,
            .y = 20,
            .z = 30,
        },
        a,
    );
}

test "vector unscale" {
    const a = Vector3(f32){
        .x = 1,
        .y = 2,
        .z = 3,
    };

    const b = Vector3(f32){
        .x = 4,
        .y = 5,
        .z = 6,
    };

    expectVectorEquals(
        Vector3(f32){
            .x = 0.25,
            .y = 0.4,
            .z = 0.5,
        },
        a.unscale(b),
    );
}

test "vector unscale_" {
    var a = Vector3(f32){
        .x = 1,
        .y = 2,
        .z = 3,
    };

    const b = Vector3(f32){
        .x = 4,
        .y = 5,
        .z = 6,
    };

    a.unscale_(b);

    expectVectorEquals(
        Vector3(f32){
            .x = 0.25,
            .y = 0.4,
            .z = 0.5,
        },
        a,
    );
}

test "vector sunscale" {
    const a = Vector3(f32){
        .x = 1,
        .y = 2,
        .z = 3,
    };

    expectVectorEquals(
        Vector3(f32){
            .x = 0.1,
            .y = 0.2,
            .z = 0.3,
        },
        a.sunscale(10),
    );
}

test "vector sunscale_" {
    var a = Vector3(f32){
        .x = 1,
        .y = 2,
        .z = 3,
    };

    a.sunscale_(10);

    expectVectorEquals(
        Vector3(f32){
            .x = 0.1,
            .y = 0.2,
            .z = 0.3,
        },
        a,
    );
}

test "vector eq" {
    const a = Vector3(f32){
        .x = 1,
        .y = 2,
        .z = 3,
    };

    const b = Vector3(f32){
        .x = 1,
        .y = 2,
        .z = 3,
    };

    const c = Vector3(f32){
        .x = 1,
        .y = 3,
        .z = 2,
    };

    t.expect(a.eq(b));
    t.expect(!a.eq(c));
}

test "vector dot" {
    const a = Vector2(f32){
        .x = -6,
        .y = 8
    };

    const b = Vector2(f32){
        .x = 5,
        .y = 12,
    };

    t.expectEqual(
        @as(f32, 66),
        a.dot(b),
    );
}

test "vector magnitude2" {
    const a = Vector2(f32){
        .x = 3,
        .y = 4,
    };

    t.expectEqual(
        @as(f32, 25),
        a.magnitude2(),
    );
}

test "vector magnitude" {
    const a = Vector2(f32){
        .x = 3,
        .y = 4,
    };

    t.expectEqual(
        @as(f32, 5),
        a.magnitude(),
    );
}

test "vector cross" {
    const a = Vector3(f32){
        .x = 2,
        .y = 3,
        .z = 4,
    };

    const b = Vector3(f32){
        .x = 5,
        .y = 6,
        .z = 7,
    };

    expectVectorEquals(
        Vector3(f32){
            .x = -3,
            .y = 6,
            .z = -3,
        },
        a.cross(b),
    );

    const c = Vector2(f32){
        .x = 2,
        .y = 3,
    };

    const d = Vector2(f32){
        .x = 5,
        .y = 6,
    };

    t.expectEqual(
        @as(f32, -3),
        c.cross(d),
    );
}

test "vector cross_" {
    var a = Vector3(f32){
        .x = 2,
        .y = 3,
        .z = 4,
    };

    const b = Vector3(f32){
        .x = 5,
        .y = 6,
        .z = 7,
    };

    a.cross_(b);

    expectVectorEquals(
        Vector3(f32){
            .x = -3,
            .y = 6,
            .z = -3,
        },
        a,
    );
}

test "vector extend" {
    const a = Vector2(f32){
        .x = 1,
        .y = 2,
    };

    const b = a.extend(3);

    expectVectorEquals(
        Vector3(f32){
            .x = 1,
            .y = 2,
            .z = 3,
        },
        b,
    );

    const c = b.extend(4);

    expectVectorEquals(
        Vector4(f32){
            .x = 1,
            .y = 2,
            .z = 3,
            .w = 4,
        },
        c,
    );
}

test "vector truncate" {
    const a = Vector4(f32){
        .x = 1,
        .y = 2,
        .z = 3,
        .w = 4,
    };

    const b = a.truncate();

    expectVectorEquals(
        Vector3(f32){
            .x = 1,
            .y = 2,
            .z = 3,
        },
        b,
    );

    const c = b.truncate();

    expectVectorEquals(
        Vector2(f32){
            .x = 1,
            .y = 2,
        },
        c,
    );
}

test "vector projectOn" {
    const a = Vector2(f32){
        .x = 1,
        .y = 2,
    };

    const b = Vector2(f32){
        .x = 3,
        .y = 4,
    };

    expectVectorApproxEq(
        Vector2(f32){
            .x = 1.32,
            .y = 1.76,
        },
        a.projectOn(b),
        0.001,
    );
}

test "vector projectOn_" {
    var a = Vector2(f32){
        .x = 1,
        .y = 2,
    };

    const b = Vector2(f32){
        .x = 3,
        .y = 4,
    };

    a.projectOn_(b);

    expectVectorApproxEq(
        Vector2(f32){
            .x = 1.32,
            .y = 1.76,
        },
        a,
        0.001,
    );
}

test "vector normalize" {
    const a = Vector3(f32){
        .x = 3,
        .y = 1,
        .z = 2,
    };

    expectVectorApproxEq(
        Vector3(f32){
            .x = 0.802,
            .y = 0.267,
            .z = 0.534,
        },
        a.normalize(),
        0.01,
    );
}

test "vector normalize_" {
    var a = Vector3(f32){
        .x = 3,
        .y = 1,
        .z = 2,
    };

    a.normalize_();

    expectVectorApproxEq(
        Vector3(f32){
            .x = 0.802,
            .y = 0.267,
            .z = 0.534,
        },
        a,
        0.01,
    );
}

test "vector normalizeTo" {
    const a = Vector3(f32){
        .x = 3,
        .y = 1,
        .z = 2,
    };

    expectVectorApproxEq(
        Vector3(f32){
            .x = 8.02,
            .y = 2.67,
            .z = 5.34,
        },
        a.normalizeTo(10),
        0.01,
    );
}

test "vector normalizeTo_" {
    var a = Vector3(f32){
        .x = 3,
        .y = 1,
        .z = 2,
    };

    a.normalizeTo_(10);

    expectVectorApproxEq(
        Vector3(f32){
            .x = 8.02,
            .y = 2.67,
            .z = 5.34,
        },
        a,
        0.01,
    );
}

test "vector angle" {
    const a = Vector3(f32){
        .x = 1,
        .y = 2,
        .z = 3,
    };

    const b = Vector3(f32){
        .x = 4,
        .y = 5,
        .z = 6,
    };

    t.expectApproxEqRel(@as(f32, 0.22573), a.angle(b), std.math.sqrt(std.math.epsilon(f32)));

    const c = Vector2(f32){
        .x = 1,
        .y = 2,
    };

    const d = Vector2(f32){
        .x = -3,
        .y = 4,
    };

    t.expectApproxEqRel(@as(f32, 1.1071), c.angle(d), std.math.sqrt(std.math.epsilon(f32)));
}
