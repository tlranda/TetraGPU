import numpy as np
import tqdm

import argparse

class fakeProgress():
    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable
    def __iter__(self):
        for obj in self.iterable:
            yield obj
    def refresh(self, *args, **kwargs):
        return None
    def update(self, *args, **kwargs):
        return None

def allocate(c):
    return np.zeros(c*4, dtype=int), np.zeros(c*6, dtype=int)

def init(c):
    ain, aout = allocate(c)
    for i in range(c):
        hi_mask = i << 4
        ain[(i*4):((i+1)*4)] = [hi_mask | n for n in [0x7, 0xB, 0xD, 0xE]]
        #print(f"Cell {i}, mask={hi_mask}, input={ain[(i*4):((i+1)*4)]}")
        aout[(i*6)  ] = ain[(i*4)  ] & ain[(i*4)+1]
        aout[(i*6)+1] = ain[(i*4)+1] & ain[(i*4)+2]
        aout[(i*6)+2] = ain[(i*4)+2] & ain[(i*4)+3]
        aout[(i*6)+3] = ain[(i*4)  ] & ain[(i*4)+2]
        aout[(i*6)+4] = ain[(i*4)+1] & ain[(i*4)+3]
        aout[(i*6)+5] = ain[(i*4)  ] & ain[(i*4)+3]
        #print(f"Expect output={aout[(i*6):((i+1)*6)]}")
    return ain, aout

def bitwiseAND_64(shuffle0, shuffle1, shuffle2, shuffle3, laneIDs):
    # In python, this looks like:
    out = np.zeros((6,len(shuffle0)), dtype=int) # len represents active subwarps
    out[0,:] = np.array([a & b for (a,b) in zip(shuffle0, shuffle1)])
    out[1,:] = np.array([a & b for (a,b) in zip(shuffle1, shuffle2)])
    out[2,:] = np.array([a & b for (a,b) in zip(shuffle2, shuffle3)])
    out[3,:] = np.array([a & b for (a,b) in zip(shuffle0, shuffle2)])
    out[4,:] = np.array([a & b for (a,b) in zip(shuffle1, shuffle3)])
    out[5,:] = np.array([a & b for (a,b) in zip(shuffle0, shuffle3)])
    # In CUDA, it's closer to this
    for (subwarp_idx, (v0, v1, v2, v3, lID)) in enumerate(
            zip(shuffle0, shuffle1, shuffle2, shuffle3, laneIDs)
            ):
        left_op = v0
        right_op = v3
        if lID == 1 or lID == 2 or lID == 4:
            if lID == 2:
                left_op = v2
            else:
                left_op = v1
        if lID == 0 or lID == 1 or lID == 3:
            if lID == 0:
                right_op = v1
            else:
                right_op = v2
        assert out[lID,subwarp_idx] == left_op & right_op
    return out

def kernel(blocks, threads, ain, c, aout, args):
    progressclass = fakeProgress if args.no_progress else tqdm.tqdm
    thread_ids = []
    print(f"Identify immediate-exit threads and per-thread info for {blocks*threads} threads")
    progress = progressclass(total=blocks*threads, miniters=1)
    for block in range(blocks):
        prev_threads = len(thread_ids)
        block_announce_ldp = False
        for thread in range(threads):
            warp = thread // 32
            tid = (block * threads) + thread
            wid = tid % 32
            lid = wid % 6
            ldp = 3*(((tid // 32)*5) + (wid // 6))
            if ldp >= c:
                if not block_announce_ldp:
                    block_announce_ldp = True
                    print(f"Block {block} thread {thread} early-exits for LDP indicating cell >= {c}")
                progress.update(threads-thread)
                progress.refresh()
                continue
            elif wid > 29:
                #print(f"Block {block} thread {thread} early-exits due to WID {wid} > 29")
                progress.update(2)
                progress.refresh()
                continue
            thread_ids.append((block,warp,thread,tid,wid,lid,ldp))
            progress.update(1)
        print(f"Block {block} adds {len(thread_ids)-prev_threads} threads")
        progress.refresh()
    progress.refresh()
    thread_ids = np.atleast_2d(thread_ids)

    reads, writes = dict(), dict()

    global_read = dict()

    # Step all threads up through global reads
    print(f"Perform all (max 2) global reads for all active threads")
    progress = progressclass(total=len(thread_ids), miniters=1)
    for (block, warp, thread, threadID, warpID, laneID, laneDepth) in thread_ids:
        # The final subwarp (c-laneDepth < 3) can have special behaviors based
        # on how many unrolls it actually needs to perform.
        # If the number of cells is divisible by 3, there is no special
        # behavior required, checked by #cells-laneDepth == 2.
        # If the difference #cells-laneDepth == 1, the second read should only
        # be performed by the lowest-2 threads to complete the second element
        # and not read the non-existent third.
        # If the difference == 0, there is no second read and the highest-2
        # threads should not perform their first read either.

        read_indicator = c-laneDepth-1
        # FIRST READ
        if read_indicator >= 1 or\
                read_indicator == 0 and laneID < 4:
            reads[threadID] = [(laneDepth*4)+laneID]
            global_read[threadID] = [ain[reads[threadID][0]]]

        # SECOND READ
        if read_indicator > 1 or\
                read_indicator == 1 and laneID < 2:
            reads[threadID] += [(laneDepth*4)+laneID+6]
            global_read[threadID] += [ain[reads[threadID][1]]]
        progress.update(1)
    progress.refresh()

    # Ensure every element is read exactly once
    if args.skip_read_validation:
        print(f"Skipping read authentication")
    else:
        print(f"Ensuring elements are all read exactly once")
        read_idx = sorted([idx for thread_read in reads.values() for idx in thread_read])
        should_read_idx = set(range(len(ain)))
        missing_reads = should_read_idx.difference(set(read_idx))
        duplicated_reads = [_ for _ in set(read_idx) if read_idx.count(_) > 1]
        assert len(missing_reads) == 0
        assert len(duplicated_reads) == 0

    # Step all threads through algorithm here similar to a simple GPU scheduler
    # Iterate and peel the block column first
    print(f"Simulate threads block-by-block")
    for block_id in tqdm.tqdm(sorted(set(thread_ids[:,0])), total=blocks, miniters=1, position=0):
        block_where = np.where(thread_ids[:,0] == block_id)[0]
        block = thread_ids[block_where,1:]
        # Iterate and peel the warp column
        for warp_id in progressclass(sorted(set(block[:,0])), total=(len(block)+31)//32, miniters=1, position=1):
            warp_where = np.where(block[:,0] == warp_id)[0]
            warp = block[warp_where,1:]

            # Warp-synchronous from here, don't go below explicitly
            # thread,threadID,warpID,laneID,laneDepth
            threadIDs  = warp[:,1]
            warpIDs    = warp[:,2]
            laneIDs    = warp[:,3]
            laneDepths = warp[:,4]

            # All lane depths within a sub-warp are the same, but not all lane
            # depths within a warp are the same

            # Re-fetch the read information for our threads in the warp
            global_reads = np.asarray([global_read[tid][0] for tid in threadIDs if tid in global_read.keys()])

            # Emulate shuffles -- in python we actually get all of them at once
            # CUDA: out_base_index = laneDepth * 18
            # CUDA: __shfl_sync(0xfffffffc, global_read, [0-3], 6)
            # CUDA: __syncthreads()
            shuffle0 = np.asarray([global_reads[i] for i in warpIDs if laneIDs[i] == 0], dtype=int)
            shuffle1 = np.asarray([global_reads[i] for i in warpIDs if laneIDs[i] == 1], dtype=int)
            shuffle2 = np.asarray([global_reads[i] for i in warpIDs if laneIDs[i] == 2], dtype=int)
            shuffle3 = np.asarray([global_reads[i] for i in warpIDs if laneIDs[i] == 3], dtype=int)
            # EVERY SUBWARP PARTICIPATES IN THIS OPERATION
            # CUDA: bitwiseAND_64(shuffle[0-3], laneID, out)
            out = bitwiseAND_64(shuffle0, shuffle1, shuffle2, shuffle3, laneIDs)
            #print(block_id, warp_id, out)
            """
            read_indicator = c-laneDepth-1
            # FIRST READ
            if read_indicator >= 1 or\
                    read_indicator == 0 and laneID < 4:
                reads[threadID] = [(laneDepth*4)+laneID]
                global_read[threadID] = [ain[reads[threadID][0]]]

            # SECOND READ
            if read_indicator > 1 or\
                    read_indicator == 1 and laneID < 2:
            """

            # Early-exit condition for 1-unroll on final subwarp
            read_indicator = c-laneDepths-1
            alive = np.where(read_indicator > 1)[0]
            threadIDs  = threadIDs[alive]
            warpIDs    = warpIDs[alive]
            laneIDs    = laneIDs[alive]
            laneDepths = laneDepths[alive]
            # Move output: out+=6
            # CUDA: __shfl_sync(0xfffffffc, global_read, [4-5], 6)
            # CUDA: __syncthreads()
            shuffle0 = np.asarray([global_reads[i] for i in warpIDs if laneIDs[i] == 4], dtype=int)
            shuffle1 = np.asarray([global_reads[i] for i in warpIDs if laneIDs[i] == 5], dtype=int)
            # CUDA: // Do second read here if you don't do both at the top
            global_reads = np.asarray([global_read[tid][1] for tid in threadIDs if tid in global_read.keys()])
            # CUDA: __shfl_sync(0xfffffffc, global_read, [0-1], 6)
            # CUDA: __syncthreads()
            shuffle2 = np.asarray([global_reads[i] for i in warpIDs if laneIDs[i] == 0], dtype=int)
            shuffle3 = np.asarray([global_reads[i] for i in warpIDs if laneIDs[i] == 1], dtype=int)
            # CUDA: bitwiseAND_64(shuffle[0-3], laneID, out)
            out = bitwiseAND_64(shuffle0, shuffle1, shuffle2, shuffle3, laneIDs)
            #print(block_id, warp_id, out)
            # Early-exit condition for 2-unroll on final subwarp
            read_indicator = c-laneDepths-1
            alive = np.where(read_indicator > 2)[0]
            threadIDs  = threadIDs[alive]
            warpIDs    = warpIDs[alive]
            laneIDs    = laneIDs[alive]
            laneDepths = laneDepths[alive]
            # Move output: out+=6
            # CUDA: __shfl_sync(0xfffffffc, glboal_read, [2-5], 6)
            # CUDA: __syncthreads()
            shuffle0 = np.asarray([global_reads[i] for i in warpIDs if laneIDs[i] == 2], dtype=int)
            shuffle1 = np.asarray([global_reads[i] for i in warpIDs if laneIDs[i] == 3], dtype=int)
            shuffle2 = np.asarray([global_reads[i] for i in warpIDs if laneIDs[i] == 4], dtype=int)
            shuffle3 = np.asarray([global_reads[i] for i in warpIDs if laneIDs[i] == 5], dtype=int)
            # CUDA: bitwiseAND_64(shuffle[0-3], laneID, out)
            out = bitwiseAND_64(shuffle0, shuffle1, shuffle2, shuffle3, laneIDs)
            #print(block_id, warp_id, out)

def prekernel(args):
    c = args.N_CELLS
    print(f"Prepare kernel for {c} cells")
    ain, aout = init(c)

    # Max 1024 threads per block in 32-thread warps
    # If every warp exits 2 threads, you miss out on 64 threads, so 960 usable
    # threads per block. Every warp of 30 threads is sub-divided into 5 subwarps
    # of 6 threads, giving 32x5 = 160 subwarp groups. Every subwarp group unrolls
    # 3 cells, so every block can therefore fully unroll at most 480 cells


    # ACTUAL CALCUATION -- SHOULD NOT BE WRONG
    # Every cell requires 6 threads, but every sextet unrolls up to 3 cells
    threads_needed = 6 * ((c+2)//3)
    print(f"Raw thread demand with unrolling: {threads_needed}")
    # For every warp (32 threads) we launch, we early-exit 2 threads
    threads_needed = ((threads_needed+29)//30)*32
    print(f"FullWarp thread-demand with unrolling+early exit: {threads_needed}")

    # Every block has no more than 1024 threads due to HW architecture
    N_THREADS = 1024
    # This means we can process up to 480 cells in a single block
    # 6*((480+2)//3)    == 6*(482//3)   == 6*160 == 960
    # ((960+29)//30)*32 == (989//30)*32 == 32*32 == 1024
    # Increasing to 481 requires 966 raw threads which adds a warp (33 warps),
    # ergo we require another block to allocate that warp of threads.
    #
    # Memory demand per cell is 4 vertices from TV and an unknown number of
    # edges in VE, however the expectation is #edges per vertex will resemble
    # the mean vertex degree
    # The TV array accesses are coalesced; if tetras generally have vertices
    # with similar IDs and nearby tetras have nearby vertex IDs, then the VE
    # accesses are not strictly coalesced but should fall into L2 without great
    # difficulty

    N_BLOCKS = (threads_needed+1023)//1024
    print(f"FullWarp requires launch {N_BLOCKS} blocks of {N_THREADS} threads")

    # SHORTCUT CALCULATION -- BASED ON LIMITS ABOVE
    N_THREADS = 1024
    CELLS_PER_BLOCK = 480
    N_BLOCKS = (c+CELLS_PER_BLOCK-1)//CELLS_PER_BLOCK
    print(f"Shortcut calcuation: {N_BLOCKS} block of {N_THREADS} threads")

    kernel(N_BLOCKS, N_THREADS, ain, c, aout, args)

if __name__ == '__main__':
    prs = argparse.ArgumentParser()
    prs.add_argument('N_CELLS', nargs='?', type=int, default=4, help="Number of cells to simulate (default: %(default)s)")
    prs.add_argument('--skip-read-validation', action='store_true', help="Read validation takes a lot of memory/time in Python; worth skipping for very large simulations (default: %(default)s)")
    prs.add_argument('--no-progress', action='store_true', help="Disable progress bars (default: %(default)s)")

    args = prs.parse_args()
    prekernel(args)

