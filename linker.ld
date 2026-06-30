/* Titan LLM Kernel — bare-metal aarch64 linker script
 * Load address: 0x40080000 (standard AArch64 Linux Image load point,
 * reused here for QEMU virt machine compatibility)
 * Model weights mapped at: 0x80000000 (2GB physical — above kernel)
 */

OUTPUT_FORMAT("elf64-littleaarch64")
OUTPUT_ARCH(aarch64)
ENTRY(_start)

MEMORY {
    /* 512MB kernel region */
    RAM (rwx) : ORIGIN = 0x40080000, LENGTH = 512M
    /* 4GB model weight region — GGUF loaded here by VMM */
    MODEL (r)  : ORIGIN = 0x80000000, LENGTH = 4096M
}

SECTIONS {
    . = 0x40080000;

    /* ARM64 Linux Image header — needed for QEMU -kernel flag */
    .head.text : {
        KEEP(*(.head.text))
    } > RAM

    .text : {
        *(.text._start)
        *(.text*)
    } > RAM

    . = ALIGN(4096);
    .rodata : {
        *(.rodata*)
    } > RAM

    . = ALIGN(4096);
    .data : {
        *(.data*)
    } > RAM

    . = ALIGN(4096);
    .bss (NOLOAD) : {
        __bss_start = .;
        *(.bss*)
        *(COMMON)
        __bss_end = .;
    } > RAM

    . = ALIGN(4096);
    /* Stack: 8MB */
    __stack_top = . + 0x800000;

    . = ALIGN(4096);
    /* Heap: 64MB for kernel allocator */
    __heap_start = .;
    __heap_end  = __heap_start + 0x4000000;

    /* Model weights symbol — physical address of GGUF region */
    __model_base = ORIGIN(MODEL);
    __model_size = LENGTH(MODEL);

    /DISCARD/ : {
        *(.comment)
        *(.note*)
        *(.eh_frame*)
    }
}
