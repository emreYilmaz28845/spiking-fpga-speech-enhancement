/* platform.c – minimal cache helpers for Standalone on Zynq‑7000 */

#include "platform.h"
#include "xil_cache.h"

void init_platform(void)
{
    /* enable caches (UART has already been initialised by the FSBL) */
    Xil_ICacheEnable();
    Xil_DCacheEnable();
}

void cleanup_platform(void)
{
    /* flush & disable caches before handing control back to FSBL/U‑Boot */
    Xil_DCacheDisable();
    Xil_ICacheDisable();
}
