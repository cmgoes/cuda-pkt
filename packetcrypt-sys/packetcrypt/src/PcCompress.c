/**
 * (C) Copyright 2019
 * Caleb James DeLisle
 *
 * SPDX-License-Identifier: (LGPL-2.1-only OR LGPL-3.0-only)
 */
#include "PcCompress.h"
#include "Util.h"

#include <stdlib.h>
#include <assert.h>
#include <string.h>

// consensus-critical
static uint64_t pathForNum(uint64_t num, int branchHeight) {
    return Util_reverse64(num) >> (64 - branchHeight);
}

// consensus-critical
static PcCompress_Entry_t* getEntryByIndex(PcCompress_t* tbl, uint16_t num) {
    assert(num < tbl->count);
    PcCompress_Entry_t* e = &tbl->entries[num];
    return e;
}

__globla__ bool mkEntries2Kernal (PcCompress_t* d_tbl, uint64_t* d_annNumbers, PcCompress_Entry_t* d_e, uint16_t* d_nextFree, 
uint64_t mask, uint64_t bits, uint16_t flags, uint16_t iDepth, uint64_t annCount) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    uint16_t eNum = *d_nextFree;
    if (!((annNumbers[index] ^ bits) & mask)) {
        d_e->flags = flags | PcCompress_F_COMPUTABLE;

        if (flags & PcCompress_F_LEAF && bits == d_annNumbers[index]) {
            // this entry IS an announcement
            d_e->childLeft = UINT16_MAX;
            d_e->childRight = UINT16_MAX;
            return 0;
        }
        Util_BUG_IF(flags & PcCompress_F_LEAF);

        d_e->childLeft = *d_nextFree;
        Util_BUG_IF(mkEntries2(d_tbl, d_annNumbers, bits, iDepth - 1, eNum, d_nextFree, annCount));

        d_e->childRight = *d_nextFree;
        uint64_t nextBits = bits | (((uint64_t)1) << (iDepth - 1));
        Util_BUG_IF(mkEntries2(d_tbl, d_annNumbers, nextBits, iDepth - 1, eNum, d_nextFree, annCount));

        if (d_tbl->entries[d_e->childRight].flags & PcCompress_F_PAD_ENTRY) {
            d_tbl->entries[d_e->childLeft].flags |= PcCompress_F_PAD_SIBLING;
        }
        return 0;
    }
}

// consensus-critical
PcCompress_Entry_t* PcCompress_getRoot(PcCompress_t* tbl) { return getEntryByIndex(tbl, 0); }

__global__ bool mkEntriesKernal (PcCompress_t* d_tbl, uint64_t* d_annPaths, PcCompress_Entry_t* d_e, uint16_t* d_nextFree,
uint64_t mask, uint64_t bits, uint16_t depth, uint64_t pathCount, uint64_t annCount, uint16_t right) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    uint16_t eNum = *d_nextFree;
    if (!((annPaths[index] ^ bits) & mask)) { 
        // This entry is a parent of an announcement
        if (depth == d_tbl->branchHeight && bits == d_annPaths[index]) {
            // this entry IS an announcement
            d_e->childLeft = UINT16_MAX;
            d_e->childRight = UINT16_MAX;
            d_e->flags = right | PcCompress_F_LEAF | PcCompress_F_COMPUTABLE;
            return;
        }
        assert(depth != d_tbl->branchHeight);

        d_e->childLeft = *d_nextFree;
        mkEntries(d_tbl, d_annPaths, bits, depth+1, d_nextFree, eNum, pathCount, annCount, 0);
        d_e->childRight = *d_nextFree;
        uint64_t nextBits = bits | (((uint64_t)1) << depth);
        mkEntries(d_tbl, d_annPaths, nextBits, depth+1, d_nextFree,
            eNum, pathCount, annCount, PcCompress_F_RIGHT);

        d_e->flags = right | PcCompress_F_COMPUTABLE;
        if (d_tbl->entries[d_e->childRight].flags & PcCompress_F_PAD_ENTRY) {
            d_tbl->entries[d_e->childLeft].flags |= PcCompress_F_PAD_SIBLING;
        }
        if (!(bits & mask)) {
            d_e->flags |= PcCompress_F_FIRST_ENTRY;
        }
        return;
    }
} 
// consensus-critical
PcCompress_Entry_t* PcCompress_getAnn(PcCompress_t* tbl, uint64_t annNum) {
    uint64_t path = pathForNum(annNum, tbl->branchHeight);
    PcCompress_Entry_t* e = PcCompress_getRoot(tbl);
    for (int i = 0; i < tbl->branchHeight; i++) {
        uint16_t next = (path & 1) ? e->childRight : e->childLeft;
        e = getEntryByIndex(tbl, next);
        path >>= 1;
    }
    assert(e->flags & PcCompress_F_LEAF);
    return e;
}

// consensus-critical
PcCompress_Entry_t* PcCompress_getParent(PcCompress_t* tbl, PcCompress_Entry_t* e) {
    if (e->parent >= tbl->count) {
        assert(e->parent == UINT16_MAX);
        assert(e == tbl->entries);
        return NULL;
    }
    return getEntryByIndex(tbl, e->parent);
}

// consensus-critical
PcCompress_Entry_t* PcCompress_getSibling(PcCompress_t* tbl, PcCompress_Entry_t* e) {
    uint16_t num = (e - tbl->entries);
    PcCompress_Entry_t* p = PcCompress_getParent(tbl, e);
    if (!p) { return NULL; }
    uint16_t sib = (p->childLeft == num) ? p->childRight : p->childLeft;
    assert((p->childLeft == num) ? 1 : (p->childRight == num));
    return getEntryByIndex(tbl, sib);
}

static int mkEntries2(
    PcCompress_t* tbl,
    const uint64_t annNumbers[static PacketCrypt_NUM_ANNS],
    uint64_t bits,
    uint16_t iDepth,
    uint16_t parentNum,
    uint16_t* nextFree,
    uint64_t annCount
) {
    uint16_t eNum = *nextFree;
    Util_BUG_IF(eNum >= tbl->count);
    *nextFree = eNum + 1;
    PcCompress_Entry_t* e = &tbl->entries[eNum];
    e->parent = parentNum;

    uint64_t mask = UINT64_MAX << iDepth;

    uint16_t flags = 0;
    flags |= ((bits >> iDepth) & 1) ? PcCompress_F_RIGHT : 0;
    flags |= (iDepth == 0) ? PcCompress_F_LEAF : 0;
    flags |= ((bits & mask) == 0) ? PcCompress_F_FIRST_ENTRY : 0;

    PcCompress_t* d_tbl;
    uint64_t* d_annNumbers;
    PcCompress_Entry_t* d_e;
    uint16_t* d_nextFree;

    cudaMalloc((void **)&d_tbl, sizeof(PcCompress_t));
    cudaMalloc((void **)&d_annNumbers, PacketCrypt_NUM_ANNS*sizeof(uint64_t));
    cudaMalloc((void **)&d_e, sizeof(PcCompress_Entry_t));
    cudaMalloc((void **)&d_nextFree, sizeof(uint16_t));

    cudaMemcpy(d_tbl, tbl, sizeof(PcCompress_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_annNumbers, annNumbers, PacketCrypt_NUM_ANNS*sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_e, e, sizeof(PcCompress_Entry_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nextFree, nextFree, sizeof(uint16_t), cudaMemcpyHostToDevice);

    mkEntries2Kernal<<<1, PacketCrypt_NUM_ANNS>>>(d_tbl, d_annNumbers, d_e, d_nextFree, mask, bits, flags, iDepth, annCount);
    cudaMemcpy(tbl, d_tbl, sizeof(PcCompress_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(annNumbers, d_annNumbers, PacketCrypt_NUM_ANNS*sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(e, d_e, sizeof(PcCompress_Entry_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(nextFree, d_nextFree, sizeof(uint16_t), cudaMemcpyDeviceToHost);
    cudaFree(d_tbl);
    cudaFree(d_annNumbers);
    cudaFree(d_e);
    cudaFree(d_nextFree);    

    // Not the parent of any announcement
    e->childLeft = UINT16_MAX;
    e->childRight = UINT16_MAX;

    if (bits >= annCount) {
        // pad entry
        Util_BUG_IF(!(flags & PcCompress_F_RIGHT));
        e->flags = flags | PcCompress_F_PAD_ENTRY |
            PcCompress_F_HAS_HASH | PcCompress_F_HAS_RANGE | PcCompress_F_HAS_START;
        Buf_OBJSET(&e->e, 0xff);
        return 0;
    }

    // it's a sibling for which data must be provided
    e->flags = flags;
    return 0;
}

PcCompress_t* PcCompress_mkEntryTable2(
    uint64_t annCount,
    const uint64_t annNumbers[static PacketCrypt_NUM_ANNS]
) {
    for (int i = 0; i < PacketCrypt_NUM_ANNS; i++) {
        if (annNumbers[i] >= annCount) { return NULL; }
    }
    int branchHeight = Util_log2ceil(annCount);
    int capacity = branchHeight * PacketCrypt_NUM_ANNS * 3;
    PcCompress_t* out = calloc(sizeof(PcCompress_t) + sizeof(PcCompress_Entry_t) * capacity, 1);
    assert(out);
    out->count = capacity;
    out->branchHeight = branchHeight;
    uint16_t nextFree = 0;
    if (mkEntries2(out, annNumbers, 0, branchHeight, UINT16_MAX, &nextFree, annCount)) {
        free(out);
        return NULL;
    }
    out->count = nextFree;
    return out;
}

// consensus-critical
static void mkEntries(
    PcCompress_t* tbl,
    const uint64_t annPaths[static PacketCrypt_NUM_ANNS],
    uint64_t bits,
    int depth,
    uint16_t* nextFree,
    uint16_t parentNum,
    uint64_t pathCount,
    uint64_t annCount,
    uint16_t right
) {
    uint16_t eNum = *nextFree;
    assert(eNum < tbl->count);
    *nextFree = eNum + 1;
    PcCompress_Entry_t* e = &tbl->entries[eNum];
    e->parent = parentNum;

    uint64_t mask = (1ull << depth) - 1;

    PcCompress_t* d_tbl;
    uint64_t* d_annPaths;
    PcCompress_Entry_t* d_e;
    uint16_t* d_nextFree;

    cudaMalloc((void **)&d_tbl, sizeof(PcCompress_t));
    cudaMalloc((void **)&d_annPaths, PacketCrypt_NUM_ANNS*sizeof(uint64_t));
    cudaMalloc((void **)&d_e, sizeof(PcCompress_Entry_t));
    cudaMalloc((void **)&d_nextFree, sizeof(uint16_t));

    cudaMemcpy(d_tbl, tbl, sizeof(PcCompress_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_annPaths, annPaths, PacketCrypt_NUM_ANNS*sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_e, e, sizeof(PcCompress_Entry_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nextFree, nextFree, sizeof(uint16_t), cudaMemcpyHostToDevice);

    mkEntriesKernal<<<1, PacketCrypt_NUM_ANNS>>>(d_tbl, d_annPaths, d_e, d_nextFree, mask, bits, flags, depth, pathCount, annCount, right);
    cudaMemcpy(tbl, d_tbl, sizeof(PcCompress_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(annPathss, d_annPaths, PacketCrypt_NUM_ANNS*sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(e, d_e, sizeof(PcCompress_Entry_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(nextFree, d_nextFree, sizeof(uint16_t), cudaMemcpyDeviceToHost);
    cudaFree(d_tbl);
    cudaFree(d_annPaths);
    cudaFree(d_e);
    cudaFree(d_nextFree); 
    

    // Not a parent of an announcement
    e->childLeft = UINT16_MAX;
    e->childRight = UINT16_MAX;

    if (pathForNum(bits, tbl->branchHeight) >= annCount) {
        assert(right);
        // It's a pad entry
        e->flags = right | PcCompress_F_PAD_ENTRY |
            PcCompress_F_HAS_HASH | PcCompress_F_HAS_RANGE | PcCompress_F_HAS_START;
        if (depth == tbl->branchHeight) { e->flags |= PcCompress_F_LEAF; }
        Buf_OBJSET(&e->e, 0xff);
        return;
    }

    // it's a sibling for which data must be provided
    e->flags = right;
    if (depth == tbl->branchHeight) {
        e->flags |= PcCompress_F_LEAF;
    }
    if (!(bits & mask)) {
        e->flags |= PcCompress_F_FIRST_ENTRY;
    }
    return;
}

// consensus-critical
PcCompress_t* PcCompress_mkEntryTable(
    uint64_t annCount,
    const uint64_t annNumbers[static PacketCrypt_NUM_ANNS]
) {
    for (int i = 0; i < PacketCrypt_NUM_ANNS; i++) {
        if (annNumbers[i] >= annCount) { return NULL; }
    }
    int branchHeight = Util_log2ceil(annCount);
    uint64_t pathCount = pathForNum(annCount, branchHeight);

    const uint64_t annPaths[] = {
        pathForNum(annNumbers[0], branchHeight), pathForNum(annNumbers[1], branchHeight),
        pathForNum(annNumbers[2], branchHeight), pathForNum(annNumbers[3], branchHeight)
    };

    int capacity = branchHeight * PacketCrypt_NUM_ANNS * 3;
    PcCompress_t* out = calloc(sizeof(PcCompress_t) + sizeof(PcCompress_Entry_t) * capacity, 1);
    assert(out);
    out->count = capacity;
    out->branchHeight = branchHeight;
    uint16_t nextFree = 0;
    mkEntries(out, annPaths, 0, 0, &nextFree, UINT16_MAX, pathCount, annCount, 0);
    out->count = nextFree;

    PcCompress_t* out2 = PcCompress_mkEntryTable2(annCount, annNumbers);
    assert(out->count == out2->count);
    assert(!memcmp(out, out2, sizeof(PcCompress_t) + sizeof(PcCompress_Entry_t) * out->count));
    free(out);
    return out2;
}

bool PcCompress_hasExplicitRange(PcCompress_Entry_t* e)
{
    // right leaf needs an explicit range provided at the beginning
    if ((e->flags & (PcCompress_F_LEAF | PcCompress_F_RIGHT | PcCompress_F_PAD_ENTRY)) ==
        (PcCompress_F_LEAF | PcCompress_F_RIGHT))
    {
        return true;
    }

    // anything that is not a LEAF
    // not a COMPUTABLE
    // not a PAD_ENTRY nor a sibling of one
    return (!(e->flags & (
        PcCompress_F_LEAF |
        PcCompress_F_COMPUTABLE |
        PcCompress_F_PAD_ENTRY | PcCompress_F_PAD_SIBLING)));
}
