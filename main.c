/*====================================================================
 *  snn_dma_demo  –  main.c  (TIMED POLLING VERSION, VER‑2025‑07‑23‑b)
 *  ------------------------------------------------------------------
 *  Adds cycle‑accurate timing and a compile‑time verbosity switch.
 *===================================================================*/

#include <stdio.h>
#include "xparameters.h"
#include "xil_printf.h"
#include "xil_cache.h"
#include "xaxidma.h"
#include "xaxidma_hw.h"
#include "ff.h"              /* FatFs */
#include "platform.h"        /* init_platform / cleanup_platform */
#include "xiltimer.h"         /* XTime, XTime_GetTime, COUNTS_PER_SECOND */

/* ------------------------------------------------------------------ */
/*  Verbosity switch                                                  */
/*      PRINT_PER_FRAME = 0  → banner + errors + summary only         */
/*      PRINT_PER_FRAME = 1  → also show one status line per frame    */
/* ------------------------------------------------------------------ */
#ifndef PRINT_PER_FRAME
#define PRINT_PER_FRAME 0
#endif

#if PRINT_PER_FRAME
  #define FRAME_PRINTF(...)  xil_printf(__VA_ARGS__)
#else
  #define FRAME_PRINTF(...)
#endif
/* Always‑on macros for errors and final summary -------------------- */
#define ERR_PRINTF(...)     xil_printf(__VA_ARGS__)
#define INFO_PRINTF(...)    xil_printf(__VA_ARGS__)   /* banner + summary */

/* ------------------------------------------------------------------ */
/*  Build‑time parameters                                             */
/* ------------------------------------------------------------------ */
#define WORDS_PER_FRAME   9
#define BYTES_PER_FRAME   (WORDS_PER_FRAME * 4)
#define BUF_DEPTH         512
#define DMA_TIMEOUT       1000000     /* ~10 ms @ 100 MHz            */

#define DMA_DEV_ID     0U             /* only one AXI‑DMA           */

/* Frame buffers (aligned to cache line) ---------------------------- */
static u32 in_buf [BUF_DEPTH][WORDS_PER_FRAME] __attribute__((aligned(64)));
static u32 out_buf[BUF_DEPTH][WORDS_PER_FRAME] __attribute__((aligned(64)));

static XAxiDma AxiDma;

/* ------------------------------------------------------------------ */
/*  Timing accumulators (64‑bit cycle counters)                       */
/* ------------------------------------------------------------------ */
static XTime cycles_total_start;      /* wall‑clock start */
static u64   cycles_fpga      = 0;    /* DMA kick → idle  */
static u64   cycles_cpu_prep  = 0;    /* cache ops        */
static u64   cycles_sd_read   = 0;    /* f_read()         */
static u64   cycles_sd_write  = 0;    /* f_write()        */

/* Convert cycles to µs --------------------------------------------- */
static inline u64 cyc_to_us(u64 cyc)
{
    return (cyc * 1000000ULL) / (u64)COUNTS_PER_SECOND;
}

/* ------------------------------------------------------------------ */
/*  Helper: wait until both DMA channels idle (with timeout)          */
/* ------------------------------------------------------------------ */
static int wait_dma_idle(void)
{
    int guard = DMA_TIMEOUT;
    while (guard-- && (   XAxiDma_Busy(&AxiDma, XAXIDMA_DMA_TO_DEVICE)
                       || XAxiDma_Busy(&AxiDma, XAXIDMA_DEVICE_TO_DMA)))
        ;
    return guard > 0 ? 0 : -1;
}

/* ------------------------------------------------------------------ */
/*  Send one frame through the SNN core                               */
/* ------------------------------------------------------------------ */
static int snn_run_frame(u32 *in, u32 *out, int idx)
{
    XTime t_p0, t_p1, t_fpga0, t_fpga1, t_p2;

    /* -- CPU pre‑DMA: cache flushes -------------------------------- */
    XTime_GetTime(&t_p0);
    Xil_DCacheFlushRange((UINTPTR)in,  BYTES_PER_FRAME);
    Xil_DCacheFlushRange((UINTPTR)out, BYTES_PER_FRAME);
    XTime_GetTime(&t_p1);

    /* -- Kick DMA (MM2S first, tiny gap, then S2MM) ---------------- */
    t_fpga0 = t_p1;
    XAxiDma_SimpleTransfer(&AxiDma, (UINTPTR)in,  BYTES_PER_FRAME,
                           XAXIDMA_DMA_TO_DEVICE);
    for (volatile int d = 0; d < 100; ++d) ;
    XAxiDma_SimpleTransfer(&AxiDma, (UINTPTR)out, BYTES_PER_FRAME,
                           XAXIDMA_DEVICE_TO_DMA);

    if (wait_dma_idle() == 0) {
        XTime_GetTime(&t_fpga1);
        Xil_DCacheInvalidateRange((UINTPTR)out, BYTES_PER_FRAME);
        XTime_GetTime(&t_p2);

        cycles_fpga     += (u64)(t_fpga1 - t_fpga0);
        cycles_cpu_prep += (u64)(t_p1 - t_p0) + (u64)(t_p2 - t_fpga1);

        FRAME_PRINTF("[%d] TX ok   RX ok\r\n", idx);
        return 0;
    }

    /* -- timeout: dump status registers ---------------------------- */
    XTime_GetTime(&t_fpga1);                 /* stop timing anyway    */
    cycles_fpga     += (u64)(t_fpga1 - t_fpga0);
    cycles_cpu_prep += (u64)(t_p1 - t_p0);

    u32 txsr = XAxiDma_ReadReg(AxiDma.RegBase,
                               XAXIDMA_TX_OFFSET + XAXIDMA_SR_OFFSET);
    u32 rxsr = XAxiDma_ReadReg(AxiDma.RegBase,
                               XAXIDMA_RX_OFFSET + XAXIDMA_SR_OFFSET);
    ERR_PRINTF("[%d] TIMEOUT  TX_SR=0x%08x  RX_SR=0x%08x\r\n",
               idx, txsr, rxsr);
    return -1;
}

/* ------------------------------------------------------------------ */
/*  Minimal DMA initialisation (simple mode, no interrupts)           */
/* ------------------------------------------------------------------ */
static int dma_init(void)
{
    XAxiDma_Config *Cfg = XAxiDma_LookupConfig(DMA_DEV_ID);
    if (!Cfg) return XST_FAILURE;
    if (XAxiDma_CfgInitialize(&AxiDma, Cfg) != XST_SUCCESS) return XST_FAILURE;
    if (XAxiDma_HasSg(&AxiDma)) return XST_FAILURE;
    return XST_SUCCESS;
}

/* ------------------------------------------------------------------ */
/*  main                                                              */
/* ------------------------------------------------------------------ */
int main(void)
{
    FATFS fatfs;
    FIL   fin, fout;
    UINT  br, bw;
    FRESULT fr;

    init_platform();
    INFO_PRINTF("\nSNN DMA demo (timed polling) starting…\r\n");

    XTime_GetTime(&cycles_total_start);

    if (dma_init() != XST_SUCCESS) {
        ERR_PRINTF("DMA init failed\r\n");
        goto fail;
    }

    fr = f_mount(&fatfs, "0:", 1);
    if (fr != FR_OK) { ERR_PRINTF("f_mount %d\r\n", fr); goto fail; }

    fr = f_open(&fin, "input.bin", FA_READ);
    if (fr != FR_OK) { ERR_PRINTF("open input %d\r\n", fr); goto unmount; }

    fr = f_open(&fout, "output.bin", FA_CREATE_ALWAYS | FA_WRITE);
    if (fr != FR_OK) { ERR_PRINTF("open output %d\r\n", fr); goto close_in; }

    /* -- Process file in BUF_DEPTH‑frame chunks -------------------- */
    int frame_total = 0;
    do {
        XTime t0, t1;

        XTime_GetTime(&t0);
        fr = f_read(&fin, in_buf, sizeof in_buf, &br);
        XTime_GetTime(&t1);
        cycles_sd_read += (u64)(t1 - t0);

        if (fr != FR_OK) { ERR_PRINTF("read err %d\r\n", fr); break; }

        int frames = br / BYTES_PER_FRAME;
        for (int f = 0; f < frames; ++f) {
            if (snn_run_frame(in_buf[f], out_buf[f], frame_total) != 0) {
                ERR_PRINTF("Aborting after frame %d\r\n", frame_total);
                goto close_out;
            }
            ++frame_total;
        }

        XTime_GetTime(&t0);
        fr = f_write(&fout, out_buf, frames * BYTES_PER_FRAME, &bw);
        XTime_GetTime(&t1);
        cycles_sd_write += (u64)(t1 - t0);

        if (fr != FR_OK || bw != frames * BYTES_PER_FRAME) {
            ERR_PRINTF("write err %d\r\n", fr); break; }

    } while (br == sizeof in_buf);

    INFO_PRINTF("Done. %d frames processed.\r\n", frame_total);

close_out:
    f_close(&fout);
close_in:
    f_close(&fin);
unmount:
    f_mount(0, "0:", 0);

    /* -- Timing summary ------------------------------------------- */
    {
        XTime cycles_total_end;
        XTime_GetTime(&cycles_total_end);
        u64 cycles_total = (u64)(cycles_total_end - cycles_total_start);

        u64 cycles_other = cycles_total;
        if (cycles_other > cycles_fpga)      cycles_other -= cycles_fpga;
        if (cycles_other > cycles_cpu_prep)  cycles_other -= cycles_cpu_prep;
        if (cycles_other > cycles_sd_read)   cycles_other -= cycles_sd_read;
        if (cycles_other > cycles_sd_write)  cycles_other -= cycles_sd_write;

        INFO_PRINTF("\r\n=== Timing summary ===\r\n");
        INFO_PRINTF("Total wall time    : %8llu us\r\n",
                    (unsigned long long)cyc_to_us(cycles_total));
        INFO_PRINTF("  FPGA (DMA+SNN)   : %8llu us  (avg %llu us/frame)\r\n",
                    (unsigned long long)cyc_to_us(cycles_fpga),
                    frame_total ? (unsigned long long)
                      cyc_to_us(cycles_fpga / frame_total) : 0ULL);
        INFO_PRINTF("  CPU prep/cleanup : %8llu us  (avg %llu us/frame)\r\n",
                    (unsigned long long)cyc_to_us(cycles_cpu_prep),
                    frame_total ? (unsigned long long)
                      cyc_to_us(cycles_cpu_prep / frame_total) : 0ULL);
        INFO_PRINTF("  SD read          : %8llu us\r\n",
                    (unsigned long long)cyc_to_us(cycles_sd_read));
        INFO_PRINTF("  SD write         : %8llu us\r\n",
                    (unsigned long long)cyc_to_us(cycles_sd_write));
        INFO_PRINTF("  Other            : %8llu us\r\n",
                    (unsigned long long)cyc_to_us(cycles_other));

        u64 payload_bytes = (u64)frame_total * BYTES_PER_FRAME;
        u64 t_payload_us  = cyc_to_us(cycles_fpga + cycles_cpu_prep);
        if (t_payload_us == 0) t_payload_us = 1;
        u64 bytes_per_s = (payload_bytes * 1000000ULL) / t_payload_us;
        INFO_PRINTF("\r\nPayload throughput : %llu B/s (~%llu KB/s)\r\n",
                    (unsigned long long)bytes_per_s,
                    (unsigned long long)(bytes_per_s / 1024ULL));
    }

fail:
    cleanup_platform();
    return 0;
}
