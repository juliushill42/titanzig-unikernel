// kernel/uart.zig
// PL011 UART — QEMU virt machine base address 0x09000000
// This is our only output mechanism. No printf. No OS. Just MMIO.

const UART_BASE: usize = 0x09000000;

const DR   = @as(*volatile u32, @ptrFromInt(UART_BASE + 0x000)); // Data Register
const FR   = @as(*volatile u32, @ptrFromInt(UART_BASE + 0x018)); // Flag Register
const IBRD = @as(*volatile u32, @ptrFromInt(UART_BASE + 0x024)); // Integer Baud Rate
const CR   = @as(*volatile u32, @ptrFromInt(UART_BASE + 0x030)); // Control Register

const FR_TXFF: u32 = 1 << 5; // TX FIFO full

pub fn init() void {
    CR.* = 0;           // disable UART
    IBRD.* = 26;        // 115200 baud @ 48MHz
    CR.* = (1 << 8) | (1 << 9) | 1; // TXE | RXE | UARTEN
}

pub fn putchar(c: u8) void {
    while (FR.* & FR_TXFF != 0) {}
    DR.* = c;
}

pub fn puts(s: []const u8) void {
    for (s) |c| {
        if (c == '\n') putchar('\r');
        putchar(c);
    }
}

pub fn print(comptime fmt: []const u8, args: anytype) void {
    var buf: [512]u8 = undefined;
    const s = @import("std").fmt.bufPrint(&buf, fmt, args) catch buf[0..0];
    puts(s);
}

pub fn hex(val: u64) void {
    const digits = "0123456789abcdef";
    puts("0x");
    var i: i32 = 60;
    while (i >= 0) : (i -= 4) {
        putchar(digits[(val >> @intCast(i)) & 0xF]);
    }
}
