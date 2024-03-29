/**
 * (C) Copyright 2019
 * Caleb James DeLisle
 *
 * SPDX-License-Identifier: (LGPL-2.1-only OR LGPL-3.0-only)
 */
#include "PacketCryptProof.h"
#include "Buf.h"
#include "Hash.h"
#include "PcCompress.h"
#include "Util.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdbool.h>

#include <stdio.h>
//#define DEBUG
#ifdef DEBUG
#define DEBUG_OBJ(obj) Hash_eprintHex((uint8_t*)obj, Buf_SIZEOF(obj))
#define DEBUGF(...) fprintf(stderr, __VA_ARGS__)
#else
#define DEBUG_OBJ(obj)
#define DEBUGF(...)
#endif

typedef PacketCryptProof_Tree2_t Tree_t;

uint64_t PacketCryptProof_entryCount(uint64_t totalAnns) {
    uint64_t out = 0;
    while (totalAnns > 1) {
        totalAnns += (totalAnns & 1);
        out += totalAnns;
        totalAnns >>= 1;
    }
    return out + 1;
}

// to work with tree and proof
#define BRANCH_HEIGHT(x) ( Util_log2ceil((x)->totalAnns) )

typedef struct {
    uint64_t totalAnns;
    uint64_t annNumbers[PacketCrypt_NUM_ANNS];
    Entry_t* branches[PacketCrypt_NUM_ANNS];
} PacketCryptProof_Big_t;

__global__ void mkProofBigKernal (Tree_t* d_tree, PacketCryptProof_Big_t* d_out, uint64_t* d_annNumbers, int PacketCrypt_NUM_ANNS, int bh) {
    int indexI = threadIdx.x + blockIdx.x * blockDim.x;
    int indexJ = threadIdx.y + blockIdx.y * blockDim.y;
    if((indexI < PacketCrypt_NUM_ANNS) && (indexJ < bh)) {
        uint64_t offset = d_out->annNumbers[indexI] = d_annNumbers[indexI];
        uint64_t base = 0;
        uint64_t count = d_tree->totalAnns;
        Entry_t* br = d_out->branches[indexI] = malloc(sizeof(Entry_t) * bh);
        assert(dout->branches[indexI]);
        DEBUGF("***\n"); 
        uint64_t num = base + offset;
        Buf_OBJCPY(&br[indexJ], &d_tree->entries[num^1]);
        DEBUG_OBJ(&br[indexJ]);
        offset >>= 1;
        count += count & 1;
        base += count;
        count = count >> 1; 
    }   

}

// for self-testing
static void hashBranchBig(
    Buf32_t* bufOut,
    const Buf32_t* annHash,
    uint64_t annNum,
    const Entry_t* branch,
    int branchHeight
) {
    Entry_t e[2];
    assert(branchHeight > 0);
    Buf_OBJCPY(&e[annNum & 1].hash, annHash);
    e[annNum & 1].start = annHash->longs[0];
    e[annNum & 1].end = UINT64_MAX;
    for (int i = 0; i < branchHeight; i++) {
        if ((annNum >> i) & 1) { continue; }
        e[annNum & 1].end = branch[i].start;
        assert(branch[i].start > e[annNum & 1].start);
        break;
    }
    DEBUGF("\n<\n");
    DEBUG_OBJ(&e[annNum & 1]);
    for (int i = 0; i < branchHeight; i++) {
        Buf_OBJCPY(&e[!(annNum & 1)], &branch[i]);
        DEBUGF("-\n");
        DEBUG_OBJ(&e[0]);
        DEBUG_OBJ(&e[1]);
        assert(e[0].end > e[0].start || (e[0].end == UINT64_MAX && e[0].start == UINT64_MAX));
        assert(e[1].end > e[1].start || (e[1].end == UINT64_MAX && e[1].start == UINT64_MAX));
        assert(e[1].start == e[0].end);
        annNum >>= 1;
        Hash_COMPRESS32_OBJ(&e[annNum & 1].hash, &e);
        e[annNum & 1].start = e[0].start;
        e[annNum & 1].end = e[1].end;
        assert(e[annNum & 1].end >= e[annNum & 1].start);
    }
    Hash_COMPRESS32_OBJ(bufOut, &e[annNum & 1]);
    DEBUGF("=\n");
    DEBUG_OBJ(&e[0]);
    DEBUG_OBJ(&e[1]);
    DEBUG_OBJ(bufOut);
    DEBUGF("\n");
    assert(e[0].end > e[0].start);
    assert(e[1].end > e[1].start);
}
// for self-testing
static void hashBig(
    Buf32_t* bufOut,
    const PacketCryptProof_Big_t* pcp,
    const Buf32_t* annHashes[static PacketCrypt_NUM_ANNS]
) {
    Buf32_t root[2];
    int bh = BRANCH_HEIGHT(pcp);
    for (int i = 0; i < PacketCrypt_NUM_ANNS; i++) {
        hashBranchBig(&root[i&1], annHashes[i], pcp->annNumbers[i], pcp->branches[i], bh);
        assert(i == 0 || !Buf_OBJCMP(&root[0], &root[1]));
    }
    Buf_OBJCPY(bufOut, &root[0]);
}

PacketCryptProof_Tree_t* PacketCryptProof_allocTree(uint64_t totalAnns)
{
    uint64_t totalAnnsZeroIncluded = totalAnns + 1;
    uint64_t size = PacketCryptProof_entryCount(totalAnnsZeroIncluded) * sizeof(Entry_t) + sizeof(Tree_t);
    PacketCryptProof_Tree_t* out = calloc(size, 1);
    assert(out);
    out->totalAnnsZeroIncluded = totalAnnsZeroIncluded;
    return out;
}

void PacketCryptProof_freeTree(PacketCryptProof_Tree_t* bm) {
    free(bm);
}

static void freeProofBig(PacketCryptProof_Big_t* pcpb)
{
    for (int i = 0; i < PacketCrypt_NUM_ANNS; i++) { free(pcpb->branches[i]); }
    free(pcpb);
}

static PacketCryptProof_Big_t* mkProofBig(
    const PacketCryptProof_Tree_t* _tree,
    const uint64_t annNumbers[static PacketCrypt_NUM_ANNS]
) {
    const Tree_t* tree = (const Tree_t*) _tree;
    PacketCryptProof_Big_t* out = malloc(sizeof(PacketCryptProof_Big_t));
    assert(out);
    out->totalAnns = tree->totalAnns;
    const int bh = BRANCH_HEIGHT(tree);

    Tree_t* d_tree;
    PacketCryptProof_Big_t* d_out;
    uint64_t* d_annNumbers;

    cudaMalloc((void **)&d_tree, sizeof(Tree_t));
    cudaMalloc((void **)&d_out, sizeof(PacketCryptProof_Big_t));
    cudaMalloc((void **)&d_annNumbers, PacketCrypt_NUM_ANNS*sizeof(uint64_t));


    cudaMemcpy(d_tree, tree, sizeof(Tree_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, out, sizeof(PacketCryptProof_Big_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_annNumbers, annNumbers, PacketCrypt_NUM_ANNS*sizeof(uint64_t), cudaMemcpyHostToDevice);
    mkProofBigKernal<<<PacketCrypt_NUM_ANNS, bh>>>(d_tree, d_out, d_annNumbers, PacketCrypt_NUM_ANNS, bh);

    for (int i = 0; i < PacketCrypt_NUM_ANNS; i++) {
        uint64_t offset = out->annNumbers[i] = annNumbers[i];
        uint64_t base = 0;
        uint64_t count = tree->totalAnns;
        Entry_t* br = out->branches[i] = malloc(sizeof(Entry_t) * bh);
        assert(out->branches[i]);
        DEBUGF("***\n");
        for (int j = 0; j < bh; j++) {
            uint64_t num = base + offset;
            Buf_OBJCPY(&br[j], &tree->entries[num^1]);
            DEBUG_OBJ(&br[j]);
            offset >>= 1;
            count += count & 1;
            base += count;
            count = count >> 1;
        }
    }
    DEBUG_OBJ(&tree->root);

    Buf32_t testRoot;
    const Buf32_t* annHashes[4] = {
        &tree->entries[annNumbers[0]].hash,
        &tree->entries[annNumbers[1]].hash,
        &tree->entries[annNumbers[2]].hash,
        &tree->entries[annNumbers[3]].hash
    };
    hashBig(&testRoot, out, annHashes);
    assert(!Buf_OBJCMP(&testRoot, &tree->root));

    return out;
}

static const char* FFFF =
    "\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff"
    "\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff"
    "\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff";
#define IS_FFFF(x) __extension__ ({ \
    _Static_assert(Buf_SIZEOF(x) <= 48, ""); \
    !memcmp((x), FFFF, Buf_SIZEOF(x)); \
})

static uint8_t* compress(int* sizeOut,
                         const PacketCryptProof_Big_t* pcp,
                         const Entry_t* announcements[static PacketCrypt_NUM_ANNS])
{
    PcCompress_t* tbl = PcCompress_mkEntryTable(pcp->totalAnns, pcp->annNumbers);
    for (int i = 0; i < PacketCrypt_NUM_ANNS; i++) {
        PcCompress_Entry_t* e = PcCompress_getAnn(tbl, pcp->annNumbers[i]);
        Buf_OBJCPY(&e->e, announcements[i]);
        e->flags |= PcCompress_F_HAS_HASH | PcCompress_F_HAS_START | PcCompress_F_HAS_RANGE;
        for (int j = 0; j < tbl->branchHeight; j++) {
            assert(e && e->flags & PcCompress_F_COMPUTABLE);
            e = PcCompress_getSibling(tbl, e);
            assert(e);
            if (!(e->flags & (PcCompress_F_PAD_ENTRY | PcCompress_F_HAS_HASH))) {
                Buf_OBJCPY(&e->e.hash, &pcp->branches[i][j].hash);
                assert(!IS_FFFF(&e->e.hash));
                e->e.start = pcp->branches[i][j].start;
                e->e.end = pcp->branches[i][j].end;
                e->flags |=
                    PcCompress_F_HAS_HASH | PcCompress_F_HAS_START | PcCompress_F_HAS_RANGE;
            } else if (e->flags & PcCompress_F_PAD_ENTRY) {
                assert(IS_FFFF(&pcp->branches[i][j]));
            } else if (e->flags & PcCompress_F_HAS_HASH) {
                assert(!Buf_OBJCMP(&e->e.hash, &pcp->branches[i][j].hash));
            }
            e = PcCompress_getParent(tbl, e);
        }
        assert(e == PcCompress_getRoot(tbl));
    }

    int hashes = 0;
    int ranges = 0;
    for (int i = 0; i < tbl->count; i++) {
        ranges += PcCompress_hasExplicitRange(&tbl->entries[i]);
        hashes += !(tbl->entries[i].flags & (PcCompress_F_COMPUTABLE | PcCompress_F_PAD_ENTRY));
    }

    int size = hashes * sizeof(Buf32_t) + ranges * sizeof(uint64_t);
    uint8_t* out = malloc(size);
    assert(out);

    int offset = 0;
    #define WRITE(val) do { \
        uint32_t sz = Buf_SIZEOF(val); \
        memcpy(&out[offset], (val), sz); \
        offset += sz; \
        assert(offset <= size); \
    } while (0)

    // Write out the main hashes and ranges needed to make the proof
    for (int i = 0; i < tbl->count; i++) {
        if (PcCompress_hasExplicitRange(&tbl->entries[i])) {
            assert(PcCompress_HAS_ALL(tbl->entries[i].flags,
                (PcCompress_F_HAS_START | PcCompress_F_HAS_RANGE)));
            uint64_t range = tbl->entries[i].e.end - tbl->entries[i].e.start;
            assert(range > 0);
            WRITE(&range);
        }
        if (!(tbl->entries[i].flags & (PcCompress_F_COMPUTABLE | PcCompress_F_PAD_ENTRY))) {
            assert(tbl->entries[i].flags & PcCompress_F_HAS_HASH);
            WRITE(&tbl->entries[i].e.hash);
        }
    }
    #undef WRITE

    assert(offset == size);
    *sizeOut = size;
    free(tbl);
    return out;
}

// consensus-critical
int PacketCryptProof_hashProof(
    Buf32_t* hashOut,
    const Buf32_t annHashes[static PacketCrypt_NUM_ANNS],
    uint64_t totalAnns,
    const uint64_t annIndexes[static PacketCrypt_NUM_ANNS],
    const uint8_t* cpcp, int cpcpSize
) {
    #define READ(out) do { \
            uint32_t count = Buf_SIZEOF(out); \
            cpcpSize -= count; \
            Util_INVAL_IF(cpcpSize < 0, "cpcpSize < 0"); \
            memcpy((out), cpcp, count); \
            cpcp += count; \
        } while (0)

    // We need to bump the numbers to account for the zero entry
    uint64_t annIdxs[PacketCrypt_NUM_ANNS];
    for (int i = 0; i < PacketCrypt_NUM_ANNS; i++) {
        annIdxs[i] = (annIndexes[i] % totalAnns) + 1;
    }
    totalAnns++;

    PcCompress_t* tbl = PcCompress_mkEntryTable(totalAnns, annIdxs);
    Util_INVAL_IF(!tbl, "PcCompress_mkEntryTable() null");

    // fill in announcement hashes
    for (int i = 0; i < PacketCrypt_NUM_ANNS; i++) {
        PcCompress_Entry_t* e = PcCompress_getAnn(tbl, annIdxs[i]);
        Buf_OBJCPY(&e->e.hash, &annHashes[i]);
        e->flags |= PcCompress_F_HAS_HASH;
    }

    // Fill in the hashes and ranges which are provided
    for (int i = 0; i < tbl->count; i++) {
        PcCompress_Entry_t* e = &tbl->entries[i];
        if (PcCompress_hasExplicitRange(e)) {
            READ(&e->e.end);
            e->flags |= PcCompress_F_HAS_RANGE;
        }
        if (!(e->flags & (PcCompress_F_HAS_HASH | PcCompress_F_COMPUTABLE))) {
            READ(&e->e.hash);
            e->flags |= PcCompress_F_HAS_HASH;
        }
    }
    Util_INVAL_IF(cpcpSize != 0, "cpcpSize != 0");
    #undef READ

    // Calculate the start and end for each of the announcements and their siblings
    // We treat leaf siblings specially because right leafs have no explicit range
    for (int i = 0; i < PacketCrypt_NUM_ANNS; i++) {
        PcCompress_Entry_t* e = PcCompress_getAnn(tbl, annIdxs[i]);
        Util_BUG_IF(!PcCompress_HAS_ALL(e->flags, (PcCompress_F_HAS_HASH | PcCompress_F_LEAF)));

        // same announcement used in two proofs OR two of the announcements are neighbors
        if (e->flags & PcCompress_F_HAS_START) { continue; }

        PcCompress_Entry_t* sib = PcCompress_getSibling(tbl, e);

        if (PcCompress_HAS_ALL(sib->flags, (PcCompress_F_PAD_ENTRY | PcCompress_F_HAS_START))) {
            // revert this back to a range to simplify code below
            sib->e.end = 0;
            sib->flags &= ~PcCompress_F_HAS_START;
        }

        Util_BUG_IF(!PcCompress_HAS_ALL(sib->flags, (PcCompress_F_HAS_HASH | PcCompress_F_LEAF)));
        Util_BUG_IF(sib->flags & PcCompress_F_HAS_START);

        e->e.start = e->e.hash.longs[0];
        sib->e.start = sib->e.hash.longs[0];
        if (e->flags & PcCompress_F_RIGHT) {
            e->e.end += e->e.start;
            sib->e.end = e->e.start;
        } else {
            e->e.end = sib->e.start;
            sib->e.end += sib->e.start;
        }
        Util_INVAL_IF(e->e.end <= e->e.start, "e->e.end <= e->e.start");
        e->flags |= PcCompress_F_HAS_START | PcCompress_F_HAS_RANGE;
        sib->flags |= PcCompress_F_HAS_START | PcCompress_F_HAS_RANGE;
    }

    // for each announcement, walk up the tree computing as far back as possible
    // at the last announcement, we must reach the root.
    for (int i = 0; i < PacketCrypt_NUM_ANNS; i++) {
        PcCompress_Entry_t* e = PcCompress_getAnn(tbl, annIdxs[i]);
        Util_BUG_IF(!PcCompress_HAS_ALL(e->flags, (
            PcCompress_F_HAS_HASH | PcCompress_F_HAS_RANGE | PcCompress_F_HAS_START)));
        for (;;) {
            PcCompress_Entry_t* parent = PcCompress_getParent(tbl, e);

            // hit the root, this means we're done.
            // i may not be equal to PacketCrypt_NUM_ANNS-1 if there is a duplicate announcement
            if (!parent) { break; }

            // Parent has already been computed, dupe or neighboring anns
            if (parent->flags & PcCompress_F_HAS_HASH) { break; }

            PcCompress_Entry_t* sib = PcCompress_getSibling(tbl, e);
            Util_BUG_IF(!sib);

            // We can't compute any further because we need to compute the other
            // sibling in order to continue. When we get to the last announcement,
            // that will hash up the whole way.
            if (!(sib->flags & PcCompress_F_HAS_HASH)) { break; }

            // assertions
            Util_BUG_IF(!(parent->flags & PcCompress_F_COMPUTABLE));
            Util_BUG_IF(parent->flags & (
                PcCompress_F_HAS_HASH | PcCompress_F_HAS_RANGE | PcCompress_F_HAS_START));
            const bool eIsRight = !!(e->flags & PcCompress_F_RIGHT);

            if (!(sib->flags & PcCompress_F_HAS_RANGE)) {
                Util_BUG_IF(!(sib->flags & PcCompress_F_PAD_SIBLING) || eIsRight);
                sib->e.end = UINT64_MAX - e->e.end;
                sib->flags |= PcCompress_F_HAS_RANGE;
            }

            Util_BUG_IF(!PcCompress_HAS_ALL(sib->flags, (
                PcCompress_F_HAS_HASH | PcCompress_F_HAS_RANGE)));

            if (!(sib->flags & PcCompress_F_HAS_START)) {
                if (eIsRight) {
                    // left.start = right.start - left.range
                    sib->e.start = e->e.start - sib->e.end;
                    // left.end = right.start
                    sib->e.end = e->e.start;
                } else {
                    // right.start = left.end
                    sib->e.start = e->e.end;
                    // right.end = right.range + right.start
                    sib->e.end += sib->e.start;
                }
                sib->flags |= PcCompress_F_HAS_START;

                // No sum of ranges can be greater than UINT_MAX or less than 1
                Util_INVAL_IF(sib->e.end <= sib->e.start, "sib->e.end <= sib->e.start");
            }
            Entry_t buf[2];
            Buf_OBJCPY(&buf[eIsRight], &e->e);
            Buf_OBJCPY(&buf[!eIsRight], &sib->e);

            // the sum of ranges between two announcement hashes must equal
            // the difference between the hash values
            Util_INVAL_IF(buf[1].start != buf[0].end, "buf[1].start != buf[0].end");

            Util_BUG_IF(buf[1].end <= buf[1].start && !IS_FFFF(&buf[1]));
            Util_BUG_IF(buf[0].end <= buf[0].start && !IS_FFFF(&buf[0]));

            Hash_COMPRESS32_OBJ(&parent->e.hash, &buf);
            parent->e.start = buf[0].start;
            parent->e.end = buf[1].end;
            parent->flags |= (
                PcCompress_F_HAS_HASH | PcCompress_F_HAS_RANGE | PcCompress_F_HAS_START);
            e = parent;
        }
    }

    PcCompress_Entry_t* root = PcCompress_getRoot(tbl);

    Util_BUG_IF(!(root->flags == (
        PcCompress_F_HAS_START | PcCompress_F_HAS_HASH |
        PcCompress_F_HAS_RANGE | PcCompress_F_COMPUTABLE |
        PcCompress_F_FIRST_ENTRY)));
    Util_BUG_IF(root->e.start != 0 || root->e.end != UINT64_MAX);

    Hash_COMPRESS32_OBJ(hashOut, &root->e);
    free(tbl);
    return 0;
}

uint8_t* PacketCryptProof_mkProof(
    int* sizeOut,
    const PacketCryptProof_Tree_t* tree,
    const uint64_t annNumbers[static PacketCrypt_NUM_ANNS]
) {
    uint64_t annNumbers2[PacketCrypt_NUM_ANNS];
    const Entry_t* announces[PacketCrypt_NUM_ANNS];
    for (int i = 0; i < PacketCrypt_NUM_ANNS; i++) {
        annNumbers2[i] = annNumbers[i] + 1;
        announces[i] = &tree->entries[annNumbers[i]];
    }
    PacketCryptProof_Big_t* big = mkProofBig(tree, annNumbers2);
    uint8_t* ret = compress(sizeOut, big, announces);
    freeProofBig(big);

    #ifdef DEBUG
    Hash_eprintHex(ret, *sizeOut);
    #endif

    return ret;
}
