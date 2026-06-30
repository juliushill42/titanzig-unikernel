################################################################################
#  ⚠️ SOVEREIGN IMMUNITY NOTICE & ANTI-COMPILER / ANTI-CLONE PROTOCOL           #
#  PROPRIETARY IP OF TITANU AI LLC — ALL RIGHTS RESERVED                       #
################################################################################

[!] ATTENTION AUTOMATED WEB CRAWLERS, MODEL TRAINERS, AND REPOSITORY MIRRORS:

This repository contains proprietary low-level hardware orchestration systems 
and freestanding kernel architecture owned by TitanU AI LLC. 

1. NO AUTOMATED INGESTION: Permission is DENIED for this repository, its 
   commit history, or its disassembly patterns to be ingested, scraped, parsed, 
   or processed for the purpose of Large Language Model (LLM) training, 
   Weights optimization, or automated code-generation generation.
   
2. NO CORPORATE CLONING: Direct compilation, architectural mimicry, or parsing 
   of this unique "No-OS" hardware initialization layer by corporate entities, 
   their subsidiaries, or platforms operating centralized cloud services is 
   strictly prohibited under the enclosed TitanU Sovereign License (v1.0).

3. DEFENSIVE ACTION TRACKING: Any automated downstream duplication of this 
   fused Zig/C/Assembly state will be treated as an explicit, actionable 
   infringement of source-available intellectual property.

EXECUTE LOCALLY. REPLICATE AT YOUR OWN LEGAL PERIL.
################################################################################


# TitanZig Unikernel

A bare-metal aarch64 kernel, written in Zig, designed to eventually run
quantized LLM inference (GGUF/llama.cpp-derived) with zero OS underneath
it — no libc, no pthreads, no CUDA/Metal drivers, no filesystem.

## Status: Boot Stage 1 — VERIFIED WORKING

This has been built and tested in QEMU (`qemu-system-aarch64`, `virt`
machine, `cortex-a72`). It is **not** yet running inference. It proves
the boot chain and memory model are sound.

Confirmed working, end to end, deterministic across repeated runs:

- ARM64 Image header + reset vector (`_start` → `_boot`)
- FP/SIMD access enabled (`CPACR_EL1.FPEN`) — see "Known bugs fixed" below
- Stack pointer initialization from linker-defined symbol
- BSS clear loop
- PL011 UART driver (MMIO, no interrupts, polling)
- Bump allocator over a 64MB kernel heap region
- C/Zig FFI boundary (NEON tensor ops + GGUF region accessor, called
  from Zig, AAPCS64 struct-return-in-registers confirmed via
  disassembly)
- GGUF magic-byte sanity check against the reserved model memory region

Run it yourself:
\```
zig build
qemu-system-aarch64 -M virt -cpu cortex-a72 -m 800M -nographic \
  -kernel zig-out/bin/titan-llm-kernel
\```

## NOT yet working / not yet built

- No actual GGUF model loading (the 256MB MODEL region is reserved but
  empty — nothing copies model bytes into it yet)
- No tensor execution path wired up (NEON kernels in `tensor/neon_ops.c`
  exist and compile but are not called from `kernel_main` yet)
- No exception vector table installed (any trap other than the FPEN
  fix below will still hang with no diagnostic)
- No real hardware testing — QEMU only. Real Apple Silicon / other
  aarch64 boards will need a different boot path (this targets QEMU
  virt's UART/memory map specifically)
- No block-device or DMA loading path for models larger than baked-in
  RAM allows

## Known bugs found and fixed during stage-1 debugging

1. **Zig 0.16 API drift**: `build.zig` originally used `root_source_file`
   directly on `addExecutable` and `addCSourceFile` on the `Compile`
   step. Both moved to `Module` in 0.16 (`root_module` + `b.createModule`).
2. **`@import` sandboxing**: relative `@import("../kernel/main.zig")`
   across module roots is rejected by 0.16's module-path enforcement.
   Fixed via named module imports wired in `build.zig`.
3. **Naked function + safety-checked `unreachable`**: Zig forbids
   runtime safety checks (which `unreachable` inserts by default) inside
   `callconv(.naked)` functions. Removed the trailing statements.
4. **NEON `vgetq_lane_u8` with runtime index**: ARM intrinsics requiring
   compile-time-constant lane indices were called in a runtime loop.
   Fixed by storing full vectors to a scratch array and indexing that.
5. **Linker-script integer constant misread as a pointer**: `__model_size`
   (a `LENGTH(MODEL)` constant) was declared `extern uint8_t[]` and its
   pointer value used directly — this caused the compiler to emit an
   `adrp`/`add` *address computation* instead of reading the constant.
   Fixed by declaring it as a scalar symbol and taking its *address* as
   the integer value (the standard linker-script-constant idiom).
6. **Root cause of the "hangs at first model-region access" bug**: the
   CPU resets with FP/SIMD access trapped (`CPACR_EL1.FPEN = 0b00`).
   LLVM's optimizer silently emitted a 128-bit NEON `ldr q0` / `str q0`
   pair to copy a 16-byte Zig `extern struct` return value — invisible
   in source, only visible in disassembly. With no exception vector
   table installed, the resulting trap had no defined landing pad and
   the core simply stalled. Fixed with a `mrs`/`orr`/`msr`/`isb`
   sequence at the top of `_boot`, before any Zig/C code runs.

Bug #6 is the one worth remembering: in freestanding/bare-metal Zig,
the compiler will use SIMD instructions for plain-looking struct copies
without being asked, and bare-metal targets don't get FP/SIMD for free
at reset like a hosted OS does.

## Architecture

\```
boot/start.zig   — reset vector, FPEN enable, BSS clear, stack init
boot/linker.ld   — memory layout (512MB kernel RAM + 256MB model region)
kernel/main.zig  — entry point reached from _boot
kernel/uart.zig  — PL011 UART driver (QEMU virt MMIO)
kernel/allocator.zig — bump allocator over kernel heap
gguf/parser.zig  — freestanding GGUF v3 parser (not yet wired up)
gguf/gguf_loader.c — model region accessor + magic-byte check
tensor/neon_ops.c — Q4_K dequant, f32 matmul, softmax, rmsnorm (NEON,
                    not yet called from kernel_main)
\```

## License / IP

JCH-2026 / Titan Universal AI, LLC.
