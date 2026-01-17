// safari_mpi.c
//
// A fully decentralized MPI simulation of "night safari" tours.
// Each MPI process is a tourist.
//
// There are P indistinguishable guides.
// Each guide leads exactly G tourists at a time.
// Therefore, at most P groups (P*G tourists) can be "on tour" concurrently.
//
// Core idea (decentralized, no central coordinator):
//  - Every tourist broadcasts a REQ (request) to join the next available tour.
//  - Every process maintains the same logical ordering of requests using Lamport clocks.
//  - Requests are sorted by (Lamport timestamp, rank) to form a global queue.
//  - The queue is partitioned into blocks of size G:
//        block 0 = positions [0 .. G-1]
//        block 1 = positions [G .. 2G-1]
//        ...
//    Only the first P blocks are allowed to start (because we have only P guides).
//  - The "leader" of a block is the first element in that block (pos % G == 0).
//  - A leader invites the other G-1 members (INVITE), waits for READY, then sends START.
//  - After finishing the tour, each tourist broadcasts REL to remove itself from the queue.
//
// Reliability updates applied (important for multi-machine runs):
//  1) All control messages use MPI_Isend (non-blocking) instead of MPI_Send.
//     Reason: across multiple machines, MPI_Send can block due to rendezvous protocol,
//     and if many processes block sending at once, no one progresses receives -> deadlock.
//  2) Reservation self-healing:
//     A tourist that accepted an INVITE sets reserved=1.
//     If later the queue changes and the tourist no longer belongs to that leader's block,
//     it drops the reservation. This prevents being "reserved forever" and stalling.
//
// Build:
//   mpicc -O2 -Wall -Wextra safari_mpi.c -o safari
//
// Run (example):
//   mpirun --hostfile machines -np 12 ./safari 2 4 1 800 1200 0.25
//
// Arguments:
//   P            = number of guides (max concurrent groups)
//   G            = group size (tourists per guide)
//   trips        = number of tours each tourist attempts
//   tour_ms      = duration of a tour in milliseconds
//   hospital_ms  = hospital duration if beaten, in milliseconds
//   beat_prob    = probability of being beaten (0..1)

#include <mpi.h>     // MPI API: MPI_Init, MPI_Send/Recv, MPI_Isend, MPI_Iprobe, MPI_Datatype, etc.
#include <stdio.h>   // Standard I/O: printf, fprintf, fflush
#include <stdlib.h>  // Memory + parsing + sorting: malloc, calloc, free, atoi, atof, qsort
#include <unistd.h>  // usleep for millisecond sleeps

// -------------------------
// Message protocol (types)
// -------------------------

// Enumeration of all message types used by our decentralized protocol.
typedef enum {
  MSG_REQ    = 1, // Request: "I want to join a tour"
  MSG_ACK    = 2, // Acknowledge: "I saw your request"
  MSG_INVITE = 3, // Invite: leader -> member "You are in my group; confirm"
  MSG_READY  = 4, // Ready: member -> leader "I confirm membership"
  MSG_START  = 5, // Start: leader -> group "Start the tour now"
  MSG_REL    = 6  // Release: "Remove my request from the queue"
} MsgType;

// A compact fixed-size message.
// We intentionally use 4 ints so we can create a simple contiguous MPI datatype.
typedef struct {
  int type;  // Message type (one of MsgType)
  int ts;    // Lamport timestamp at sender when this message was sent
  int a;     // Payload field A (meaning depends on message type)
  int b;     // Payload field B (meaning depends on message type)
} Msg;

// Field meanings by type:
//
// MSG_REQ:
//   ts = request timestamp of sender
//   a  = req_rank (sender rank)
//   b  = req_ts   (same as ts; explicitly included)
//
// MSG_ACK:
//   ts = timestamp when ACK was sent
//   a  = req_rank being acknowledged
//   b  = req_ts being acknowledged
//
// MSG_INVITE:
//   ts = timestamp when INVITE was sent
//   a  = leader_rank
//   b  = leader_req_ts (timestamp of leader's REQ; identifies the group instance)
//
// MSG_READY:
//   ts = timestamp when READY was sent
//   a  = leader_rank
//   b  = leader_req_ts
//
// MSG_START:
//   ts = timestamp when START was sent
//   a  = leader_rank
//   b  = leader_req_ts
//
// MSG_REL:
//   ts = timestamp when REL was sent
//   a  = rank_to_release
//   b  = unused (0)

// -------------------------
// Request queue structures
// -------------------------

// Per-rank information: does rank r currently request a tour, and with which timestamp?
typedef struct {
  int active; // 1 if rank has an active REQ, 0 if not
  int ts;     // timestamp of that REQ (valid only if active==1)
} ReqInfo;

// Entry used to build and sort the current queue of active requests.
typedef struct {
  int rank; // MPI rank of requester
  int ts;   // Lamport timestamp of its REQ
} Entry;

// -------------------------
// Lamport logical clock
// -------------------------

// Local Lamport clock value for THIS process.
static int lamport = 0;

// Lamport update rule on receive:
//   lamport = max(lamport, msg_ts) + 1
static void lamport_on_recv(int msg_ts) {
  if (msg_ts > lamport) lamport = msg_ts; // ensure we are at least msg_ts
  lamport += 1;                           // advance for the receive event
}

// Lamport update rule on send:
//   lamport = lamport + 1
// Returns the value to stamp the outgoing message.
static int lamport_on_send(void) {
  lamport += 1; // advance for the send event
  return lamport;
}

// Sleep helper: converts milliseconds to microseconds for usleep.
static void msleep(int ms) {
  usleep((useconds_t)ms * 1000u);
}

// Comparator for qsort: sorts Entry by (ts ascending, rank ascending).
// This creates a total order used to build a consistent queue.
static int cmp_entry(const void *p1, const void *p2) {
  const Entry *a = (const Entry*)p1; // first entry
  const Entry *b = (const Entry*)p2; // second entry

  // Primary key: Lamport timestamp.
  if (a->ts != b->ts) return (a->ts < b->ts) ? -1 : 1;

  // Tie-breaker key: rank.
  if (a->rank < b->rank) return -1;
  if (a->rank > b->rank) return 1;
  return 0;
}

// Build a sorted snapshot of the global request queue from our local reqs[] table.
// Input:
//   reqs        = per-rank active request info
//   world_size  = number of MPI processes
//   out         = buffer to write queue entries into (size >= world_size)
// Output:
//   returns n = number of active requests
//   out[0..n-1] sorted by (ts, rank)
static int build_sorted_queue(const ReqInfo *reqs, int world_size, Entry *out) {
  int n = 0; // number of active requests discovered so far

  // Copy all active requests into out[].
  for (int r = 0; r < world_size; r++) {
    if (reqs[r].active) {     // include only active requests
      out[n].rank = r;        // store rank
      out[n].ts   = reqs[r].ts;// store timestamp
      n++;                    // advance output count
    }
  }

  // Sort to obtain a globally comparable queue order.
  qsort(out, (size_t)n, sizeof(Entry), cmp_entry);

  return n; // return how many entries are valid
}

// Find position of a given rank in a sorted queue.
// Returns index if found, or -1 if not present.
static int find_pos_in_queue(const Entry *q, int n, int my_rank) {
  for (int i = 0; i < n; i++) {          // scan queue
    if (q[i].rank == my_rank) return i;  // found at position i
  }
  return -1; // not found
}

// Check if THIS process (my_rank) is the leader of a ready-to-start block.
// Conditions for "leader and ready":
//  - Have all ACKs for our own REQ (so everyone knows our request)
//  - Our REQ is still active and matches my_req_ts
//  - In the sorted queue, our position pos satisfies pos % G == 0 (start of a block)
//  - Block index grp = pos / G is < P (guide available)
//  - Block has G members currently present (pos + (G-1) < n)
static int is_leader_and_ready(
  int my_rank,                     // this process rank
  int P,                           // number of guides
  int G,                           // group size
  const ReqInfo *reqs,             // local request table
  int world_size,                  // number of processes
  int my_req_ts,                   // timestamp of our current REQ
  int have_all_acks,               // boolean: did we receive ACK from all other ranks?
  int *out_group_start,            // output: start index of our block in queue
  int *out_group_id,               // output: block index (0..)
  Entry *scratch                   // temporary array used for sorting
) {
  if (!have_all_acks) return 0; // cannot be leader if request not globally known yet

  // Must still be active and the stored timestamp must match our current request.
  if (!reqs[my_rank].active || reqs[my_rank].ts != my_req_ts) return 0;

  // Build current queue snapshot.
  int n = build_sorted_queue(reqs, world_size, scratch);

  // Find our position within that snapshot.
  int pos = find_pos_in_queue(scratch, n, my_rank);
  if (pos < 0) return 0; // not in queue => cannot lead

  // Must be first element in a block.
  if (pos % G != 0) return 0;

  // Determine which block we would lead.
  int grp = pos / G;

  // Only blocks 0..P-1 can start (only P guides).
  if (grp >= P) return 0;

  // Ensure block is complete: we need G requests in this block.
  if (pos + (G - 1) >= n) return 0;

  // Success: return block start and block id.
  *out_group_start = pos;
  *out_group_id    = grp;
  return 1;
}

// Check whether my_rank belongs to leader_rank's block (group instance defined by leader_req_ts).
// We only accept invites/starts if we are inside leader's current block in the queue.
static int member_belongs_to_leader(
  int my_rank,                     // this process rank
  int P,                           // number of guides
  int G,                           // group size
  const ReqInfo *reqs,             // local request table
  int world_size,                  // number of processes
  int leader_rank,                 // leader rank
  int leader_req_ts,               // leader's request timestamp (group instance id)
  Entry *scratch                   // temporary array used for sorting
) {
  // We must have an active request to be considered for any group.
  if (!reqs[my_rank].active) return 0;

  // Leader must still be active and must match the same request timestamp.
  if (!reqs[leader_rank].active || reqs[leader_rank].ts != leader_req_ts) return 0;

  // Build queue snapshot.
  int n = build_sorted_queue(reqs, world_size, scratch);

  // Find both positions in that queue.
  int my_pos     = find_pos_in_queue(scratch, n, my_rank);
  int leader_pos = find_pos_in_queue(scratch, n, leader_rank);
  if (my_pos < 0 || leader_pos < 0) return 0;

  // Leader must be at the start of a block.
  if (leader_pos % G != 0) return 0;

  // Leader's block must be within guide capacity.
  int grp = leader_pos / G;
  if (grp >= P) return 0;

  // Block must be complete.
  if (leader_pos + (G - 1) >= n) return 0;

  // We must be inside leader's block range.
  return (my_pos >= leader_pos && my_pos <= leader_pos + (G - 1));
}

// Simple deterministic pseudo-random in [0,1).
// Used only for simulation (sleep randomness and beating probability).
static double frand01(unsigned int *seed) {
  *seed = (*seed * 1103515245u + 12345u); // LCG step
  return (double)((*seed / 65536u) % 32768u) / 32768.0; // scale to [0,1)
}

// MPI_Send may block across machines (rendezvous), which can deadlock if many ranks
// send bursts simultaneously and stop receiving.
// MPI_Isend allows the program to keep progressing receives while sends complete.
//
// Note:
// We "fire-and-forget" because messages are small control signals.
// MPI_Request_free tells MPI we will not call MPI_Wait on this request.
static void send_async(const Msg *m, MPI_Datatype MPI_MSG, int dst) {
  MPI_Request req;                                        // request handle for non-blocking send
  MPI_Isend((void*)m, 1, MPI_MSG, dst, 0, MPI_COMM_WORLD, &req); // start non-blocking send
  MPI_Request_free(&req);                                 // detach request (MPI manages completion)
}

// -------------------------
// Main program
// -------------------------
int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);                                 // start MPI runtime

  int rank = 0;                                           // this process id
  int world = 0;                                          // total number of processes
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);                   // fill rank
  MPI_Comm_size(MPI_COMM_WORLD, &world);                  // fill world

  // Validate required argument count.
  if (argc < 7) {
    if (rank == 0) {                                      // only rank 0 prints usage to avoid spam
      fprintf(stderr,
        "Usage: %s P G trips tour_ms hospital_ms beat_prob\n"
        "Example: mpirun -np 20 %s 2 4 3 800 1200 0.25\n",
        argv[0], argv[0]
      );
    }
    MPI_Finalize();                                       // clean MPI shutdown
    return 1;                                             // exit with error
  }

  int P = atoi(argv[1]);                                  // parse number of guides
  int G = atoi(argv[2]);                                  // parse group size
  int trips = atoi(argv[3]);                              // parse number of trips per tourist
  int tour_ms = atoi(argv[4]);                            // parse tour duration
  int hospital_ms = atoi(argv[5]);                        // parse hospital duration
  double beat_prob = atof(argv[6]);                       // parse beating probability

  // Ensure P and G are meaningful.
  if (G <= 0 || P <= 0) {
    if (rank == 0) fprintf(stderr, "P and G must be > 0\n"); // print error from rank 0
    MPI_Finalize();                                       // shutdown MPI
    return 1;                                             // exit with error
  }

  // The spec says: T >= 2*G.
  // We warn if not met, because the algorithm may not progress if too small.
  if (world < 2 * G && rank == 0) {
    fprintf(stderr,
      "Warning: T (np=%d) < 2*G (%d). Spec says at least 2*G.\n",
      world, 2 * G
    );
  }

  // Create a custom MPI datatype that matches Msg layout: 4 contiguous ints.
  MPI_Datatype MPI_MSG;                                   // handle for custom datatype
  MPI_Type_contiguous(4, MPI_INT, &MPI_MSG);              // 4 ints -> one contiguous type
  MPI_Type_commit(&MPI_MSG);                               // commit type before use

  // Allocate per-rank request state array (one slot per rank).
  ReqInfo *reqs = (ReqInfo*)calloc((size_t)world, sizeof(ReqInfo)); // zero-initialized request table

  // Allocate scratch array used to build and sort queue snapshots.
  Entry *scratch = (Entry*)malloc((size_t)world * sizeof(Entry));   // temp buffer for queue sorting

  // Our own request state.
  int my_req_active = 0;                                  // do we currently have an active REQ?
  int my_req_ts = -1;                                     // timestamp of our current REQ
  int ack_count = 0;                                      // how many ACKs we collected for our REQ

  // Reservation state (member of someone else's group).
  int reserved = 0;                                       // did we accept an INVITE?
  int reserved_leader = -1;                               // which leader invited us
  int reserved_leader_ts = -1;                            // timestamp of leader's REQ (group instance id)

  // Leader state (we are leader of a block and collecting READY).
  int leader_waiting = 0;                                 // are we currently a leader waiting for READY?
  int leader_ready_count = 0;                             // number of READYs received from group members

  // Per-rank deterministic RNG seed (different for each rank).
  unsigned int rng = (unsigned int)(1234u + 99991u * (unsigned int)rank);

  // Main simulation: each rank attempts 'trips' tours.
  for (int trip = 0; trip < trips; trip++) {

    // Random delay to avoid synchronized bursts (helps reduce contention).
    msleep(50 + (int)(frand01(&rng) * 200.0));

    // Begin a new request attempt: reset per-trip local variables.
    my_req_active = 1;                                    // we are now requesting
    ack_count = 0;                                        // reset ACK counter
    reserved = 0;                                         // not reserved initially
    reserved_leader = -1;                                 // clear leader id
    reserved_leader_ts = -1;                              // clear leader ts
    leader_waiting = 0;                                   // not leader yet
    leader_ready_count = 0;                               // reset READY count

    // Create our REQ timestamp (Lamport send event).
    my_req_ts = lamport_on_send();                        // advance Lamport and store timestamp

    // Record our request locally (so we appear in our own queue snapshot).
    reqs[rank].active = 1;                                // mark ourselves active
    reqs[rank].ts = my_req_ts;                            // store our REQ timestamp

    // Build a REQ message to broadcast.
    Msg m;                                                // message struct for REQ
    m.type = MSG_REQ;                                     // set message type
    m.ts   = my_req_ts;                                   // stamp with our request timestamp
    m.a    = rank;                                        // requester rank
    m.b    = my_req_ts;                                   // request timestamp again (explicit id)

    // Broadcast REQ to all other ranks (UPDATE #1 uses send_async).
    for (int dst = 0; dst < world; dst++) {               // iterate over all ranks
      if (dst == rank) continue;                          // skip self
      send_async(&m, MPI_MSG, dst);                       // non-blocking send to avoid deadlocks
      printf("[rank %d] REQ sent to rank %d trip=%d (lamport=%d)\n", rank, dst, trip, lamport);
      fflush(stdout);                                     // flush to make logs appear promptly
    }

    int started = 0;                                      // becomes 1 when we can start the tour

    // Event loop: handle incoming messages and local decisions until started.
    while (!started) {

      int flag = 0;                                       // indicates if a message is available
      MPI_Status st;                                      // MPI status used by probe/recv

      // Drain all currently available messages without blocking.
      do {
        MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, &st); // check if any message is waiting
        if (!flag) break;                                 // none available -> exit drain loop

        Msg rcv;                                          // buffer for received message
        MPI_Recv(&rcv, 1, MPI_MSG, st.MPI_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // receive it

        lamport_on_recv(rcv.ts);                          // update Lamport clock on receive

        switch (rcv.type) {

          case MSG_REQ: {
            printf("[rank %d] REQ received trip=%d (lamport=%d)\n", rank, trip, lamport);
            fflush(stdout);

            int req_rank = rcv.a;                         // who requested
            int req_ts   = rcv.b;                         // their request timestamp

            reqs[req_rank].active = 1;                    // mark that rank as active
            reqs[req_rank].ts = req_ts;                   // store its request timestamp

            // Prepare ACK back to the sender.
            Msg ack;                                      // ACK message
            ack.type = MSG_ACK;                           // ACK type
            ack.ts   = lamport_on_send();                 // Lamport time for sending ACK
            ack.a    = req_rank;                          // acknowledge this requester's rank
            ack.b    = req_ts;                            // acknowledge this specific request timestamp

            send_async(&ack, MPI_MSG, st.MPI_SOURCE);     // non-blocking ACK send (UPDATE #1)
          } break;

          case MSG_ACK: {
            // Count ACK only if it acknowledges OUR current REQ (rank, my_req_ts).
            if (my_req_active && rcv.a == rank && rcv.b == my_req_ts) {
              ack_count++;
              printf("[rank %d] ACK COUNTED from %d trip=%d ack=%d/%d (lamport=%d)\n",
                     rank, st.MPI_SOURCE, trip, ack_count, world-1, lamport);
              fflush(stdout);
            }
            fflush(stdout);
          } break;

          case MSG_INVITE: {
            printf("[rank %d] INVITE received trip=%d (lamport=%d)\n", rank, trip, lamport);
            fflush(stdout);

            int leader = rcv.a;                           // leader rank
            int leader_ts = rcv.b;                        // leader request timestamp (group instance id)

            // Accept invite only if we are requesting and not reserved already.
            if (my_req_active && !reserved) {

              // Verify we truly belong to the leader's current block in the queue.
              if (member_belongs_to_leader(rank, P, G, reqs, world, leader, leader_ts, scratch)) {

                reserved = 1;                             // mark ourselves reserved
                reserved_leader = leader;                 // remember which leader we reserved for
                reserved_leader_ts = leader_ts;           // remember which leader instance we reserved for

                Msg ready;                                // READY message back to leader
                ready.type = MSG_READY;                   // READY type
                ready.ts   = lamport_on_send();           // Lamport time for sending READY
                ready.a    = leader;                      // leader rank
                ready.b    = leader_ts;                   // leader instance id (req timestamp)

                send_async(&ready, MPI_MSG, leader);      // non-blocking READY send (UPDATE #1)
              }
            }
          } break;

          case MSG_READY: {
            printf("[rank %d] READY received trip=%d (lamport=%d)\n", rank, trip, lamport);
            fflush(stdout);

            // Only a leader waiting for READY should count READY messages.
            if (leader_waiting && my_req_active) {

              // Count only READY messages meant for THIS leader instance.
              if (rcv.a == rank && rcv.b == my_req_ts) {
                leader_ready_count++;                     // increase READY count
              }
            }
          } break;

          case MSG_START: {
            printf("[rank %d] START received trip=%d (lamport=%d)\n", rank, trip, lamport);
            fflush(stdout);

            int leader = rcv.a;                           // leader rank
            int leader_ts = rcv.b;                        // leader instance id

            // Start if we are requesting and belong to that leader's block.
            if (my_req_active &&
                member_belongs_to_leader(rank, P, G, reqs, world, leader, leader_ts, scratch)) {
              started = 1;                                // leave event loop and begin tour
            }
          } break;

          case MSG_REL: {
            printf("[rank %d] REL received trip=%d (lamport=%d)\n", rank, trip, lamport);
            fflush(stdout);

            int rel_rank = rcv.a;                         // rank that released

            reqs[rel_rank].active = 0;                    // mark released rank as inactive
            reqs[rel_rank].ts = 0;                        // clear timestamp (not strictly needed)

            // If we were reserved to this leader and it released, drop reservation.
            if (reserved && rel_rank == reserved_leader) {
              reserved = 0;                               // clear reserved flag
              reserved_leader = -1;                       // clear leader rank
              reserved_leader_ts = -1;                    // clear leader instance
            }
          } break;

          default:
            // Unknown message type: ignore for safety.
            break;
        }
      } while (flag);

      // If we accepted an INVITE (reserved=1) but the queue changed and we no longer
      // belong to that leader's block, we must drop the reservation.
      // Otherwise we could reject all future invites forever -> system can stall.
      if (my_req_active && reserved) {                    // only if we are requesting and reserved
        if (!member_belongs_to_leader(                    // recompute membership under current queue
              rank, P, G, reqs, world,
              reserved_leader, reserved_leader_ts, scratch)) {
          reserved = 0;                                   // drop reservation
          reserved_leader = -1;                           // clear stored leader rank
          reserved_leader_ts = -1;                        // clear stored leader instance id
        }
      }

      // Determine whether we have ACK from all other processes.
      int have_all_acks = (ack_count >= (world - 1));     // must be >= because duplicates are harmless

      int group_start = -1;                               // queue index where our block starts
      int group_id = -1;                                  // id of our block

      // Leadership attempt: only if we are requesting, not reserved, and not already waiting as leader.
      if (my_req_active && !reserved && !leader_waiting) {

        // Check whether we are leader of a complete block among first P blocks.
        if (is_leader_and_ready(rank, P, G, reqs, world, my_req_ts, have_all_acks,
                                &group_start, &group_id, scratch)) {

          leader_waiting = 1;                             // enter leader state
          leader_ready_count = 0;                         // reset READY counter

          // Build the queue snapshot so we can determine members of our block.
          int n = build_sorted_queue(reqs, world, scratch);
          (void)n;                                        // suppress unused variable warning

          // Invite each other member in our block.
          for (int i = group_start; i < group_start + G; i++) {
            int member = scratch[i].rank;                 // get member rank from queue

            if (member == rank) continue;                 // don't invite self

            Msg inv;                                      // INVITE message
            inv.type = MSG_INVITE;                        // INVITE type
            inv.ts   = lamport_on_send();                 // Lamport time for INVITE send event
            inv.a    = rank;                              // leader rank
            inv.b    = my_req_ts;                         // leader instance id (our REQ timestamp)

            send_async(&inv, MPI_MSG, member);            // non-blocking INVITE send (UPDATE #1)
          }
        }
      }

      // If we are leader waiting for READY and received enough, we can START.
      if (leader_waiting) {
        if (leader_ready_count >= (G - 1)) {              // need READY from all other members

          // Re-check group validity (queue may have changed since invites).
          int ok = is_leader_and_ready(rank, P, G, reqs, world, my_req_ts, have_all_acks,
                                       &group_start, &group_id, scratch);

          if (ok) {
            // Build the queue again to get current group members.
            int n = build_sorted_queue(reqs, world, scratch);
            (void)n;

            // Send START to each member in the block (including self if desired; here we send to all ranks in block).
            for (int i = group_start; i < group_start + G; i++) {
              int member = scratch[i].rank;               // member rank

              Msg stmsg;                                  // START message
              stmsg.type = MSG_START;                     // START type
              stmsg.ts   = lamport_on_send();             // Lamport time for START send event
              stmsg.a    = rank;                          // leader rank
              stmsg.b    = my_req_ts;                     // leader instance id

              send_async(&stmsg, MPI_MSG, member);        // non-blocking START send (UPDATE #1)
            }

            started = 1;                                  // leader starts immediately too
          } else {
            // If group is not valid anymore, stop being leader and try again later.
            leader_waiting = 0;                           // exit leader state
            leader_ready_count = 0;                       // reset READY count
          }
        }
      }

      msleep(2);                                          // small sleep to reduce busy CPU spinning
    }

    // -------------------------
    // Tour simulation
    // -------------------------

    printf("[rank %d] START trip=%d (lamport=%d)\n", rank, trip, lamport); // log tour start
    fflush(stdout);                                       // flush for timely logs

    msleep(tour_ms);                                      // simulate being on tour

    double r = frand01(&rng);                             // random number for beating decision
    if (r < beat_prob) {                                  // if beaten
      printf("[rank %d] got BEATEN -> hospital %dms (lamport=%d)\n", rank, hospital_ms, lamport);
      fflush(stdout);                                     // flush log
      msleep(hospital_ms);                                // simulate hospital time
    }

    printf("[rank %d] END trip=%d (lamport=%d)\n", rank, trip, lamport); // log tour end
    fflush(stdout);                                       // flush log

    // -------------------------
    // Release: remove our request from the global queue
    // -------------------------

    reqs[rank].active = 0;                                // locally mark our request as inactive
    reqs[rank].ts = 0;                                    // clear our timestamp
    my_req_active = 0;                                    // mark we are no longer requesting

    Msg rel;                                              // REL message
    rel.type = MSG_REL;                                   // REL type
    rel.ts   = lamport_on_send();                         // Lamport time for REL send event
    rel.a    = rank;                                      // releasing rank
    rel.b    = 0;                                         // unused

    // Broadcast REL to all other ranks using non-blocking send (UPDATE #1).
    for (int dst = 0; dst < world; dst++) {
      if (dst == rank) continue;                          // skip self
      send_async(&rel, MPI_MSG, dst);                     // non-blocking REL send
    }

    msleep(10);                                           // allow some time for REL propagation
  }

  MPI_Type_free(&MPI_MSG);                                // free MPI datatype
  free(reqs);                                             // free request table memory
  free(scratch);                                          // free scratch array memory
  MPI_Finalize();                                         // finalize MPI runtime
  return 0;                                               // normal program exit
}
