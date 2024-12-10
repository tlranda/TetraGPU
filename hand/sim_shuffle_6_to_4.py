import numpy as np

N_CELLS = 4
import sys
if len(sys.argv) >= 2:
    N_CELLS = int(sys.argv[1])

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

def kernel(blocks, threads, ain, c, aout):
    thread_ids = []
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
                continue
            elif wid > 29:
                print(f"Block {block} thread {thread} early-exits due to WID {wid} > 29")
                continue
            thread_ids.append((block,warp,thread,tid,wid,lid,ldp))
        print(f"Block {block} adds {len(thread_ids)-prev_threads} threads")
    thread_ids = np.atleast_2d(thread_ids)

    reads, writes = dict(), dict()

    global_read = dict()

    # Step all threads up through global reads
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

    # Ensure every element is read exactly once
    read_idx = sorted([idx for thread_read in reads.values() for idx in thread_read])
    should_read_idx = set(range(len(ain)))
    missing_reads = should_read_idx.difference(set(read_idx))
    duplicated_reads = [_ for _ in set(read_idx) if read_idx.count(_) > 1]
    assert len(missing_reads) == 0
    assert len(duplicated_reads) == 0

    # Step all threads through algorithm here similar to a simple GPU scheduler
    # Iterate and peel the block column first
    for block_id in sorted(set(thread_ids[:,0])):
        block_where = np.where(thread_ids[:,0] == block_id)[0]
        block = thread_ids[block_where,1:]
        # Iterate and peel the warp column
        for warp_id in sorted(set(block[:,0])):
            warp_where = np.where(block[:,0] == warp_id)[0]
            warp = block[warp_where,1:]

            # Warp-synchronous from here, don't go below explicitly
            # thread,threadID,warpID,laneID,laneDepth
            threadIDs  = warp[:,1]
            warpIDs    = warp[:,2]
            laneIDs    = warp[:,3]
            laneDepths = warp[:,4]

            # Re-fetch the read information for our threads in the warp
            global_reads = np.asarray([global_read[tid][0] for tid in threadIDs if tid in global_read.keys()])

            # Emulate shuffle #1, first 4 values only per subwarp
            # CUDA: __shfl_sync(0xfffffffc, global_read, [0-3], 6)
            # CUDA: __syncthreads()
            shuffle0 = np.asarray([global_reads[i] for i in warpIDs if laneIDs[i] == 0])
            shuffle1 = np.asarray([global_reads[i] for i in warpIDs if laneIDs[i] == 1])
            shuffle2 = np.asarray([global_reads[i] for i in warpIDs if laneIDs[i] == 2])
            shuffle3 = np.asarray([global_reads[i] for i in warpIDs if laneIDs[i] == 3])

            #out_base_index = laneDepth * 18
            # CUDA: bitwiseAND_64(shuffle[0-3], laneID, out)

            # Early-exit condition for 1-unroll on final subwarp
            # CUDA: __shfl_sync(0xfffffffc, global_read, [4-5], 6)
            # CUDA: __syncthreads()
            # CUDA: // Do second read here if you don't do both at the top
            # CUDA: __shfl_sync(0xfffffffc, global_read, [0-1], 6)
            # CUDA: __syncthreads()
            # Move output: out+=6
            # CUDA: bitwiseAND_64(shuffle[0-3], laneID, out)
            # Early-exit condition for 2-unroll on final subwarp
            # CUDA: __shfl_sync(0xfffffffc, glboal_read, [2-5], 6)
            # CUDA: __syncthreads()
            # Move output: out+=6
            # CUDA: bitwiseAND_64(shuffle[0-3], laneID, out)

def prekernel(c):
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

    kernel(N_BLOCKS, N_THREADS, ain, c, aout)

prekernel(N_CELLS)

