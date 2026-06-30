// gguf/parser.zig
// Freestanding GGUF v3 parser.
// Input: raw pointer to GGUF data in memory (loaded by VMM or baked in).
// No fopen. No fread. No mmap. Direct memory reads.
// Spec: https://github.com/ggml-org/gguf

const std = @import("std");

pub const GGUF_MAGIC: u32 = 0x46554747; // "GGUF"
pub const GGUF_VERSION: u32 = 3;

pub const GGMLType = enum(u32) {
    F32     = 0,
    F16     = 1,
    Q4_0    = 2,
    Q4_1    = 3,
    Q5_0    = 6,
    Q5_1    = 7,
    Q8_0    = 8,
    Q8_1    = 9,
    Q4_K    = 12,
    Q6_K    = 14,
    _,
};

pub const MetaValueType = enum(u32) {
    UINT8   = 0,
    INT8    = 1,
    UINT16  = 2,
    INT16   = 3,
    UINT32  = 4,
    INT32   = 5,
    FLOAT32 = 6,
    BOOL    = 7,
    STRING  = 8,
    ARRAY   = 9,
    UINT64  = 10,
    INT64   = 11,
    FLOAT64 = 12,
    _,
};

pub const TensorInfo = struct {
    name:     []const u8,
    n_dims:   u32,
    dims:     [4]u64,
    dtype:    GGMLType,
    offset:   u64,        // byte offset from tensor data region start
    data:     [*]const u8, // resolved pointer after parse
    size:     usize,
};

pub const GGUFHeader = struct {
    magic:        u32,
    version:      u32,
    tensor_count: u64,
    kv_count:     u64,
};

pub const ParseError = error{
    BadMagic,
    BadVersion,
    OutOfBounds,
    UnsupportedType,
};

pub const GGUFFile = struct {
    header:       GGUFHeader,
    tensors:      []TensorInfo,
    data_base:    [*]const u8,  // start of tensor data region
    total_size:   usize,
};

// Cursor-based reader over raw memory — zero copies, zero allocations for reads
const Reader = struct {
    base: [*]const u8,
    pos:  usize,
    size: usize,

    fn read(self: *Reader, comptime T: type) ParseError!T {
        const n = @sizeOf(T);
        if (self.pos + n > self.size) return ParseError.OutOfBounds;
        const val = std.mem.readInt(T, self.base[self.pos..][0..n], .little);
        self.pos += n;
        return val;
    }

    fn readString(self: *Reader) ParseError![]const u8 {
        const len = try self.read(u64);
        if (self.pos + len > self.size) return ParseError.OutOfBounds;
        const s = self.base[self.pos .. self.pos + len];
        self.pos += len;
        return s;
    }

    fn skipMetaValue(self: *Reader, vtype: MetaValueType) ParseError!void {
        switch (vtype) {
            .UINT8, .INT8, .BOOL => self.pos += 1,
            .UINT16, .INT16      => self.pos += 2,
            .UINT32, .INT32, .FLOAT32 => self.pos += 4,
            .UINT64, .INT64, .FLOAT64 => self.pos += 8,
            .STRING => { _ = try self.readString(); },
            .ARRAY => {
                const elem_type: MetaValueType = @enumFromInt(try self.read(u32));
                const count = try self.read(u64);
                for (0..count) |_| try self.skipMetaValue(elem_type);
            },
            _ => return ParseError.UnsupportedType,
        }
    }
};

pub fn parse(
    data: [*]const u8,
    size: usize,
    tensor_buf: []TensorInfo,
) ParseError!GGUFFile {
    var r = Reader{ .base = data, .pos = 0, .size = size };

    const magic   = try r.read(u32);
    if (magic != GGUF_MAGIC) return ParseError.BadMagic;

    const version = try r.read(u32);
    if (version != GGUF_VERSION) return ParseError.BadVersion;

    const tensor_count = try r.read(u64);
    const kv_count     = try r.read(u64);

    // Skip all KV metadata
    for (0..kv_count) |_| {
        _ = try r.readString(); // key
        const vtype: MetaValueType = @enumFromInt(try r.read(u32));
        try r.skipMetaValue(vtype);
    }

    // Parse tensor infos
    const n = @min(tensor_count, tensor_buf.len);
    for (0..n) |i| {
        const name   = try r.readString();
        const n_dims = try r.read(u32);
        var dims = [4]u64{ 1, 1, 1, 1 };
        for (0..n_dims) |d| dims[d] = try r.read(u64);
        const dtype: GGMLType = @enumFromInt(try r.read(u32));
        const offset = try r.read(u64);

        tensor_buf[i] = TensorInfo{
            .name   = name,
            .n_dims = n_dims,
            .dims   = dims,
            .dtype  = dtype,
            .offset = offset,
            .data   = undefined, // resolved after alignment
            .size   = 0,
        };
    }

    // Align to 32 bytes — GGUF tensor data alignment
    r.pos = (r.pos + 31) & ~@as(usize, 31);
    const data_base = data + r.pos;

    // Resolve tensor data pointers
    for (0..n) |i| {
        tensor_buf[i].data = data_base + tensor_buf[i].offset;
        tensor_buf[i].size = tensorByteSize(tensor_buf[i].dtype, &tensor_buf[i].dims);
    }

    return GGUFFile{
        .header       = .{ .magic = magic, .version = version,
                           .tensor_count = tensor_count, .kv_count = kv_count },
        .tensors      = tensor_buf[0..n],
        .data_base    = data_base,
        .total_size   = size,
    };
}

fn tensorByteSize(dtype: GGMLType, dims: *const [4]u64) usize {
    const elements = dims[0] * dims[1] * dims[2] * dims[3];
    return switch (dtype) {
        .F32  => elements * 4,
        .F16  => elements * 2,
        .Q4_0 => (elements / 32) * 18,  // 32 weights + 2B scale
        .Q4_K => (elements / 256) * 144, // super-block Q4_K
        .Q8_0 => (elements / 32) * 34,
        else  => 0,
    };
}
