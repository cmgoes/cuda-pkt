/**
 * (C) Copyright 2019
 * Caleb James DeLisle
 *
 * SPDX-License-Identifier: (LGPL-2.1-only OR LGPL-3.0-only)
 */
#include "Announce.h"
#include "Conf.h"
#include "Hash.h"
#include "RandGen.h"
#include "RandHash.h"
#include "ValidateCtx.h"

#include <assert.h>

__global__ void memocycleKernal(Buf64_t* d_buf, buf64_t* d_tmpbuf, int bufcount, int cycles) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int p = (y - 1 + bufcount) % bufcount;
    uint32_t q = d_buf[p].ints[0] % (bufcount - 1);
    int j = (y + q) % bufcount;
    Buf64_t* mP = &d_buf[p];
    Buf64_t* mJ = &d_buf[j];
    for (int k = 0; k < 8; k++) { d_tmpbuf[0].longs[k] = mP->longs[k]; }
    for (int k = 0; k < 8; k++) { d_tmpbuf[1].longs[k] = mJ->longs[k]; }
    Hash_compress64(d_buf[y].bytes, d_tmpbuf[0].bytes, sizeof tmpbuf);    
}

static inline void memocycle(Buf64_t* buf, int bufcount, int cycles) {
    Buf64_t tmpbuf[2];
    Buf64_t* d_tmpbuf;
    buf64_t* d_buf;
    cudaMalloc((void **)&d_tmpbuf, 2*sizeof(Buf64_t));
    cudaMalloc((void **)&d_buf, bufcount*sizeof(Buf64_t));
    cudaMemcpy(d_tmpbuf, tmpbuf, 2*sizeof(Buf64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_buf, buf, bufcount*sizeof(Buf64_t), cudaMemcpyHostToDevice);
    memocycleKernal<<<cycles, bufcount>>>(d_buf, d_tmpbuf, bufcount, cycles);
    cudaMemcpy(buf, d_buf, bufcount*sizeof(Buf64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(tmpbuf, d_tmpbuf, bufcount*sizeof(Buf64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_buf);
    cudaFree(d_tmp);    
}

void Announce_mkitem(uint64_t num, CryptoCycle_Item_t* item, Buf32_t* seed) {
    Hash_expand(item->bytes, 64, seed->bytes, num);
    for (uint32_t i = 1; i < Announce_ITEM_HASHCOUNT; i++) {
        Hash_compress64(item->sixtyfours[i].bytes, item->sixtyfours[i-1].bytes, 64);
    }
    memocycle(item->sixtyfours, Announce_ITEM_HASHCOUNT, Conf_AnnHash_MEMOHASH_CYCLES);
}

int Announce_createProg(PacketCrypt_ValidateCtx_t* prog, Buf32_t* seed) {
    Hash_expand((uint8_t*)prog->progbuf, sizeof prog->progbuf, seed->bytes, 0);
    int len = RandGen_generate(prog->progbuf, seed, &prog->vars);
    if (len < 0) {
        return len;
    }
    prog->progLen = len;
    return 0;
}

int Announce_mkitem2(uint64_t num, CryptoCycle_Item_t* item,
    Buf32_t* seed, PacketCrypt_ValidateCtx_t* prog)
{
    CryptoCycle_State_t state;
    CryptoCycle_init(&state, seed, num);
    if (RandHash_interpret(prog, num, &state, 2)) { return -1; }
    CryptoCycle_makeFuzzable(&state.hdr);
    CryptoCycle_crypt(&state.hdr);
    assert(!CryptoCycle_isFailed(&state.hdr));
    Buf_OBJCPY_LDST(item, &state);
    return 0;
}
