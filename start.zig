// Titan LLM Kernel — boot/start.zig
// ARM64 reset vector. No OS. No libc. Just us and the silicon.

const std = @import("std");

// Symbols from linker script
extern var __bss_start: u8;
extern var __bss_end: u8;
extern const __stack_top: u8;

// Forward declaration of kernel entry
pub extern fn kernel_main() noreturn;

// ARM64 Linux Image header (8-byte magic for QEMU -kernel compatibility)
export fn _start() linksection(".head.text") callconv(.naked) noreturn {
    // ARM64 Image header magic: MZ + branch instruction
    asm volatile (
        \\ b _boot
        \\ .quad 0                    // image load offset
        \\ .quad 0                    // image size (filled by objcopy)
        \\ .quad 0x0a                 // flags: little-endian
        \\ .quad 0
        \\ .quad 0
        \\ .quad 0
        \\ .ascii "ARM\x64"           // magic number
        \\ .long 0                    // reserved
    );
    unreachable;
}

export fn _boot() callconv(.naked) noreturn {
    asm volatile (
        // Set stack pointer from linker symbol
        \\ adrp x0, __stack_top
        \\ add  x0, x0, :lo12:__stack_top
        \\ mov  sp, x0

        // Clear BSS
        \\ adrp x0, __bss_start
        \\ add  x0, x0, :lo12:__bss_start
        \\ adrp x1, __bss_end
        \\ add  x1, x1, :lo12:__bss_end
        \\ mov  x2, #0
        \\ 1:
        \\ cmp  x0, x1
        \\ b.ge 2f
        \\ str  x2, [x0], #8
        \\ b    1b
        \\ 2:

        // Jump to Zig kernel_main — never returns
        \\ b kernel_main
        :
        : [stack] "r" (&__stack_top),
          [bss_s] "r" (&__bss_start),
          [bss_e] "r" (&__bss_end)
        : "x0", "x1", "x2", "sp"
    );
    unreachable;
}
