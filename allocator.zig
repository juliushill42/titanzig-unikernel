// kernel/allocator.zig
// Simple bump allocator over the kernel heap region.
// No free(). Kernel lifetime = model lifetime.
// For a Q4 3B model this is ~2GB of weights + ~64MB working set.

const std = @import("std");

extern const __heap_start: u8;
extern const __heap_end: u8;

var heap_ptr: usize = 0;
var heap_initialized: bool = false;

fn init() void {
    heap_ptr = @intFromPtr(&__heap_start);
    heap_initialized = true;
}

pub fn alloc(size: usize, alignment: usize) ?[*]u8 {
    if (!heap_initialized) init();

    const heap_end = @intFromPtr(&__heap_end);
    const aligned = (heap_ptr + alignment - 1) & ~(alignment - 1);

    if (aligned + size > heap_end) return null;

    heap_ptr = aligned + size;
    return @ptrFromInt(aligned);
}

pub fn allocSlice(comptime T: type, count: usize) ?[]T {
    const bytes = @sizeOf(T) * count;
    const alignment = @alignOf(T);
    const ptr = alloc(bytes, alignment) orelse return null;
    return @as([*]T, @ptrCast(@alignCast(ptr)))[0..count];
}

pub fn used() usize {
    return heap_ptr - @intFromPtr(&__heap_start);
}

pub fn remaining() usize {
    return @intFromPtr(&__heap_end) - heap_ptr;
}

// std.mem.Allocator interface wrapper for compat with Zig stdlib structures
pub fn allocator() std.mem.Allocator {
    return std.mem.Allocator{
        .ptr = undefined,
        .vtable = &vtable,
    };
}

const vtable = std.mem.Allocator.VTable{
    .alloc = vtAlloc,
    .resize = vtResize,
    .remap = vtRemap,
    .free = vtFree,
};

fn vtAlloc(_: *anyopaque, n: usize, alignment: std.mem.Alignment, _: usize) ?[*]u8 {
    return alloc(n, alignment.toByteUnits());
}

fn vtResize(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize, _: usize) bool {
    return false; // bump allocator: no resize
}

fn vtRemap(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize, _: usize) ?[*]u8 {
    return null;
}

fn vtFree(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize) void {
    // no-op: bump allocator
}
