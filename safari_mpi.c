// safari_mpi.c
//
// A fully decentralized MPI simulation of "night safari" tours.
// Each MPI process is a tourist.
// There are P indistinguishable guides and each guide leads exactly G tourists at a time.
// At most P tours can be active concurrently (because guides are limited).
//
// The algorithm is decentralized: there is NO central coordinator process.
// We use Lamport logical clocks + a globally consistent request ordering.
// Each tourist broadcasts REQ (request) to join a tour.
// All processes maintain the same sorted request queue (by (Lamport timestamp, rank)).
// The queue is partitioned into blocks of size G:
//   positions [0..G-1] form group 0,
//   positions [G..2G-1] form group 1, etc.
// Only groups 0..P-1 are allowed to start (P guides available).
//
// Group leader is the first process in a block (position % G == 0).
// Leader invites the other G-1 members (INVITE), waits for READY, then sends START.
// After the tour (and possible hospital), each member broadcasts REL to remove its REQ.
//
// Build:
//   mpicc -O2 -Wall -Wextra safari_mpi.c -o safari
//
// Run (example):
//   mpirun -np 20 ./safari 2 4 3 800 1200 0.25
//
// Arguments:
//   P            = number of guides (max concurrent groups)
//   G            = group size (tourists per guide)
//   trips        = number of tours each tourist attempts
//   tour_ms      = duration of a tour in milliseconds
//   hospital_ms  = hospital duration if beaten, in milliseconds
//   beat_prob    = probability of being beaten (0..1)

#include <mpi.h>     // MPI API: processes, send/recv, probing, datatypes, etc.
#include <stdio.h>   // printf, fprintf
#include <stdlib.h>  // malloc, calloc, free, atoi, atof, qsort
#include <unistd.h>  // usleep

// -------------------------
// Message protocol (types)
// -------------------------

// Each message in the system has one of these types.
typedef enum {
  MSG_REQ    = 1, // "I want to join a tour" (Lamport-ordered request)
  MSG_ACK    = 2, // "I saw your REQ" (classic Lamport mutual exclusion ACK)
  MSG_INVITE = 3, // leader -> member: "You belong to my group; confirm"
  MSG_READY  = 4, // member -> leader: "I confirm; I'm ready"
  MSG_START  = 5, // leader -> group: "Start the tour now"
  MSG_REL    = 6  // "I release my request; remove me from the queue"
} MsgType;

// A compact fixed-size message.
// We keep it as 4 integers so we can define a simple MPI datatype.
typedef struct {
  int type;  // MsgType: what kind of message is this
  int ts;    // Lamport timestamp of the sender at send time (for ordering)
  int a;     // payload field A (meaning depends on message type)
  int b;     // payload field B (meaning depends on message type)
} Msg;

// Meaning of Msg fields for each message type:
//
// REQ:
//   type=MSG_REQ
//   ts  = sender's Lamport time for this request
//   a   = req_rank (who is requesting)
//   b   = req_ts   (timestamp of that request; equals ts in this implementation)
//
// ACK:
//   type=MSG_ACK
//   ts  = sender's Lamport time for the ACK send event
//   a   = req_rank being acknowledged
//   b   = req_ts being acknowledged
//
// INVITE:
//   type=MSG_INVITE
//   ts  = sender's Lamport time for invite send event
//   a   = leader_rank (who leads the group)
//   b   = leader_req_ts (timestamp of leader's REQ that defines the group instance)
//
// READY:
//   type=MSG_READY
//   ts  = sender's Lamport time for ready send event
//   a   = leader_rank
//   b   = leader_req_ts
//
// START:
//   type=MSG_START
//   ts  = sender's Lamport time for start send event
//   a   = leader_rank
//   b   = leader_req_ts
//
// REL:
//   type=MSG_REL
//   ts  = sender's Lamport time for release send event
//   a   = rank_to_release (who leaves the global request queue)
//   b   = unused (0)

// -------------------------
// Request queue structures
// -------------------------

// For each process rank r we store whether r currently has an active request
// and what Lamport timestamp that request has.
typedef struct {
  int active; // 1 if this process is currently requesting, 0 otherwise
  int ts;     // Lamport timestamp of its REQ (valid only if active==1)
} ReqInfo;

// When building a sorted queue we store (rank, ts) pairs in an array.
typedef struct {
  int rank; // MPI rank of the requester
  int ts;   // Lamport timestamp of that request
} Entry;

// -------------------------
// Lamport logical clock
// -------------------------

// Global Lamport clock for THIS process (each process has its own copy).
static int lamport = 0;

// Update Lamport clock on receiving a message that was sent at msg_ts.
// Rule: lamport = max(lamport, msg_ts) + 1
static void lamport_on_recv(int msg_ts) {
  if (msg_ts > lamport) lamport = msg_ts;
  lamport += 1;
}

// Update Lamport clock on sending a message.
// Rule: lamport = lamport + 1
// Return the new Lamport time to stamp the outgoing message.
static int lamport_on_send(void) {
  lamport += 1;
  return lamport;
}

// Sleep for ms milliseconds (helper for simulation pacing).
static void msleep(int ms) {
  usleep((useconds_t)ms * 1000u);
}

// Comparator for sorting queue entries by (ts, rank) ascending.
// This implements the global total order used by Lamport's algorithm:
// first by timestamp, then by rank to break ties.
static int cmp_entry(const void *p1, const void *p2) {
  const Entry *a = (const Entry*)p1;
  const Entry *b = (const Entry*)p2;

  // First key: Lamport timestamp.
  if (a->ts != b->ts) return (a->ts < b->ts) ? -1 : 1;

  // Tie-breaker key: rank.
  if (a->rank < b->rank) return -1;
  if (a->rank > b->rank) return 1;
  return 0;
}

// Build a sorted "global request queue" snapshot from reqs[].
// Inputs:
//   reqs       - array of ReqInfo for all ranks (local view of active requests)
//   world_size - number of processes (T)
//   out        - array buffer of size at least world_size to fill with active entries
// Output:
//   returns n = number of active requests inserted into out[]
//   out[0..n-1] is sorted by (ts, rank)
static int build_sorted_queue(const ReqInfo *reqs, int world_size, Entry *out) {
  int n = 0;

  // Copy all active requests into an array of (rank, ts).
  for (int r = 0; r < world_size; r++) {
    if (reqs[r].active) {
      out[n].rank = r;
      out[n].ts   = reqs[r].ts;
      n++;
    }
  }

  // Sort to obtain the globally agreed order (eventually consistent across processes).
  qsort(out, (size_t)n, sizeof(Entry), cmp_entry);
  return n;
}

// Find the position (index) of my_rank in the sorted queue.
// Returns:
//   index in [0..n-1] if found, or -1 if not present.
static int find_pos_in_queue(const Entry *q, int n, int my_rank) {
  for (int i = 0; i < n; i++) {
    if (q[i].rank == my_rank) return i;
  }
  return -1;
}

// Decide if THIS process can act as a leader AND the group is ready to form.
// "Leader and ready" means:
//  - I have all ACKs for my own REQ (so everyone saw my request)
//  - I am active in the request queue
//  - my position pos in the queue is the start of a block (pos % G == 0)
//  - my block index grp = pos / G is < P (so a guide is available)
//  - there are at least G requests in my block (block is complete)
//
// Outputs:
//  - out_group_start: the queue position of the leader (start of the block)
//  - out_group_id: group index (0..)
//  - returns 1 if leader+ready, else 0
static int is_leader_and_ready(
  int my_rank, int P, int G,
  const ReqInfo *reqs, int world_size,
  int my_req_ts, int have_all_acks,
  int *out_group_start, int *out_group_id, Entry *scratch
) {
  // We only consider leadership if our request is globally visible (ACKed by all).
  if (!have_all_acks) return 0;

  // Ensure we still have an active request and it matches our current request timestamp.
  // (This avoids acting on stale state after we already released or re-requested.)
  if (!reqs[my_rank].active || reqs[my_rank].ts != my_req_ts) return 0;

  // Build a sorted queue snapshot.
  int n = build_sorted_queue(reqs, world_size, scratch);

  // Find our position in that queue.
  int pos = find_pos_in_queue(scratch, n, my_rank);
  if (pos < 0) return 0; // not in queue -> cannot lead

  // Leader must be the first element of a block of size G.
  if (pos % G != 0) return 0;

  // Compute which block we lead.
  int grp = pos / G;

  // Only the first P blocks are allowed to start (because there are P guides).
  if (grp >= P) return 0;

  // Block must have enough members (G) currently present.
  // If queue has fewer than pos+G entries, the group isn't complete yet.
  if (pos + (G - 1) >= n) return 0;

  // All conditions met: group is complete and has a guide slot.
  *out_group_start = pos;
  *out_group_id    = grp;
  return 1;
}

// Decide if THIS process belongs to the leader's current group instance.
// We verify membership by recomputing the sorted queue and checking:
//  - leader is active and has the exact leader_req_ts (same group instance)
//  - leader is at a block boundary (leader_pos % G == 0)
//  - leader's block index < P (guide available)
//  - the block is complete (leader_pos + G-1 < n)
//  - my_pos lies inside leader's block range
static int member_belongs_to_leader(
  int my_rank, int P, int G,
  const ReqInfo *reqs, int world_size,
  int leader_rank, int leader_req_ts,
  Entry *scratch
) {
  // I must be active to be considered a member.
  if (!reqs[my_rank].active) return 0;

  // Leader must be active and must match the requested group instance timestamp.
  // (Otherwise INVITE/START might refer to an old leader request.)
  if (!reqs[leader_rank].active || reqs[leader_rank].ts != leader_req_ts) return 0;

  // Build current queue snapshot.
  int n = build_sorted_queue(reqs, world_size, scratch);

  // Find positions of me and the leader.
  int my_pos     = find_pos_in_queue(scratch, n, my_rank);
  int leader_pos = find_pos_in_queue(scratch, n, leader_rank);
  if (my_pos < 0 || leader_pos < 0) return 0;

  // Leader must start a block.
  if (leader_pos % G != 0) return 0;

  // Leader's block must be among the first P blocks (guide available).
  int grp = leader_pos / G;
  if (grp >= P) return 0;

  // Block must be complete.
  if (leader_pos + (G - 1) >= n) return 0;

  // Membership check: I am within leader's block range.
  return (my_pos >= leader_pos && my_pos <= leader_pos + (G - 1));
}

// A small deterministic pseudo-random generator returning [0,1).
// We use it only to randomize sleep/pobicia for simulation.
static double frand01(unsigned int *seed) {
  // Linear Congruential Generator (LCG) step.
  *seed = (*seed * 1103515245u + 12345u);

  // Convert a subset of bits into a double in [0,1).
  return (double)((*seed / 65536u) % 32768u) / 32768.0;
}

// -------------------------
// Main program
// -------------------------
int main(int argc, char **argv) {
  // Initialize MPI runtime.
  MPI_Init(&argc, &argv);

  // Obtain MPI rank (id) and world size (number of processes).
  int rank = 0, world = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);

  // Validate command-line arguments on rank 0 and exit early on error.
  if (argc < 7) {
    if (rank == 0) {
      fprintf(stderr,
        "Usage: %s P G trips tour_ms hospital_ms beat_prob\n"
        "Example: mpirun -np 20 %s 2 4 3 800 1200 0.25\n",
        argv[0], argv[0]
      );
    }
    MPI_Finalize();
    return 1;
  }

  // Parse program parameters.
  int P = atoi(argv[1]);       // number of guides (=max concurrent groups)
  int G = atoi(argv[2]);       // group size
  int trips = atoi(argv[3]);   // number of attempts/tours per tourist
  int tour_ms = atoi(argv[4]); // duration of tour simulation
  int hospital_ms = atoi(argv[5]); // duration of hospital simulation
  double beat_prob = atof(argv[6]); // probability of being beaten

  // Basic parameter sanity checks.
  if (G <= 0 || P <= 0) {
    if (rank == 0) fprintf(stderr, "P and G must be > 0\n");
    MPI_Finalize();
    return 1;
  }

  // The spec says: number of tourists must be at least 2*G.
  // We only warn (do not fail) because you may want to test smaller cases.
  if (world < 2 * G && rank == 0) {
    fprintf(stderr,
      "Warning: T (np=%d) < 2*G (%d). Spec says at least 2*G.\n",
      world, 2 * G
    );
  }

  // Define an MPI datatype corresponding to Msg (4 contiguous ints).
  // This makes MPI_Send / MPI_Recv easy and portable for this message struct.
  MPI_Datatype MPI_MSG;
  MPI_Type_contiguous(4, MPI_INT, &MPI_MSG);
  MPI_Type_commit(&MPI_MSG);

  // Allocate local shared-state arrays:
  // reqs[r] says whether rank r is requesting and its request timestamp.
  ReqInfo *reqs = (ReqInfo*)calloc((size_t)world, sizeof(ReqInfo));

  // scratch is a temporary buffer used to build and sort queue snapshots.
  Entry *scratch = (Entry*)malloc((size_t)world * sizeof(Entry));

  // ----- Local state related to OUR current request -----

  int my_req_active = 0; // 1 if we currently have an outstanding request
  int my_req_ts = -1;    // Lamport timestamp of our current request (REQ)
  int ack_count = 0;     // how many ACKs we received for (rank, my_req_ts)

  // ----- Reservation state (we accepted an INVITE as a member) -----

  int reserved = 0;              // 1 if we already reserved ourselves for a leader's group
  int reserved_leader = -1;      // which leader we reserved for
  int reserved_leader_ts = -1;   // which leader REQ timestamp defines that group instance

  // ----- Leader state (we are leader and collecting READY) -----

  int leader_waiting = 0;        // 1 if we are currently acting as leader and waiting for READY
  int leader_ready_count = 0;    // number of READY messages collected (from other members)

  // Seed for per-process pseudo-randomness (different per rank).
  unsigned int rng = (unsigned int)(1234u + 99991u * (unsigned int)rank);

  // Repeat "trip attempt" cycle trips times.
  for (int trip = 0; trip < trips; trip++) {

    // (1) Random pause before requesting, to avoid perfect synchronization.
    msleep(50 + (int)(frand01(&rng) * 200.0));

    // (2) Start a new request:
    // reset per-trip state variables.
    my_req_active = 1;
    ack_count = 0;

    reserved = 0;
    reserved_leader = -1;
    reserved_leader_ts = -1;

    leader_waiting = 0;
    leader_ready_count = 0;

    // Generate a new Lamport timestamp for our REQ and store it locally.
    my_req_ts = lamport_on_send();
    reqs[rank].active = 1;
    reqs[rank].ts = my_req_ts;

    // Build the REQ message and broadcast it to all other processes.
    Msg m;
    m.type = MSG_REQ;
    m.ts   = my_req_ts;  // stamp request with current Lamport time
    m.a    = rank;       // requester rank
    m.b    = my_req_ts;  // request timestamp (explicitly included)

    for (int dst = 0; dst < world; dst++) {
      if (dst == rank) continue; // do not send to ourselves
      MPI_Send(&m, 1, MPI_MSG, dst, 0, MPI_COMM_WORLD);
      printf("[rank %d] REQ sent to rank %d trip=%d (lamport=%d)\n", rank, dst, trip, lamport);
    }

    // started==1 means we received/triggered START for our group.
    int started = 0;

    // (3) Event loop: process messages and decide when to start.
    while (!started) {

      // (3a) Drain all currently available incoming messages without blocking.
      int flag = 0;
      MPI_Status st;

      do {
        // Probe for any incoming message (any sender, tag 0).
        MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, &st);
        if (!flag) break; // no message available right now

        // Receive the message that was just probed.
        Msg rcv;
        MPI_Recv(&rcv, 1, MPI_MSG, st.MPI_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Update Lamport clock based on received message timestamp.
        lamport_on_recv(rcv.ts);

        // Handle the message based on its type.
        switch (rcv.type) {

          case MSG_REQ: {
            printf("[rank %d] REQ received trip=%d (lamport=%d)\n", rank, trip, lamport);
            // Another process is requesting to join a tour.
            // Update our local request table so our queue snapshot includes them.
            int req_rank = rcv.a;
            int req_ts   = rcv.b;

            reqs[req_rank].active = 1;
            reqs[req_rank].ts = req_ts;

            // Send ACK back to the sender confirming we saw their REQ.
            Msg ack;
            ack.type = MSG_ACK;
            ack.ts   = lamport_on_send();
            ack.a    = req_rank; // acknowledge this requester's rank
            ack.b    = req_ts;   // acknowledge this specific request timestamp

            MPI_Send(&ack, 1, MPI_MSG, st.MPI_SOURCE, 0, MPI_COMM_WORLD);
          } break;

          case MSG_ACK: {
            printf("[rank %d] ACK received trip=%d (lamport=%d)\n", rank, trip, lamport);
            // ACK refers to a specific request (a=req_rank, b=req_ts).
            // We only count it if it acknowledges OUR current request.
            if (my_req_active && rcv.a == rank && rcv.b == my_req_ts) {
              ack_count++;
            }
          } break;

          case MSG_INVITE: {
            printf("[rank %d] INVITE received trip=%d (lamport=%d)\n", rank, trip, lamport);
            // A leader invites us to join their group instance.
            int leader = rcv.a;
            int leader_ts = rcv.b;

            // We accept only if:
            // - we are currently requesting (active),
            // - we are not already reserved for someone else,
            // - we actually belong to the leader's block in the global queue snapshot.
            if (my_req_active && !reserved) {
              if (member_belongs_to_leader(rank, P, G, reqs, world, leader, leader_ts, scratch)) {

                // Mark that we are now reserved for that leader.
                reserved = 1;
                reserved_leader = leader;
                reserved_leader_ts = leader_ts;

                // Reply with READY to confirm we will start with this group.
                Msg ready;
                ready.type = MSG_READY;
                ready.ts   = lamport_on_send();
                ready.a    = leader;
                ready.b    = leader_ts;

                MPI_Send(&ready, 1, MPI_MSG, leader, 0, MPI_COMM_WORLD);
              }
            }
          } break;

          case MSG_READY: {
            printf("[rank %d] READY received trip=%d (lamport=%d)\n", rank, trip, lamport);
            // READY is sent to a leader; only the leader cares.
            // In this implementation, READY contains (a=leader_rank, b=leader_req_ts).
            if (leader_waiting && my_req_active) {
              // Count READY only if it refers to THIS leader instance (me + my_req_ts).
              if (rcv.a == rank && rcv.b == my_req_ts) {
                leader_ready_count++;
              }
            }
          } break;

          case MSG_START: {
            printf("[rank %d] START received trip=%d (lamport=%d)\n", rank, trip, lamport);
            // Leader signals that the tour should start for its group instance.
            int leader = rcv.a;
            int leader_ts = rcv.b;

            // We start if we are currently requesting AND we belong to that leader's block.
            if (my_req_active &&
                member_belongs_to_leader(rank, P, G, reqs, world, leader, leader_ts, scratch)) {
              started = 1;
            }
          } break;

          case MSG_REL: {
            printf("[rank %d] REL received trip=%d (lamport=%d)\n", rank, trip, lamport);
            // Someone finished and releases their request; remove them from our queue view.
            int rel_rank = rcv.a;

            reqs[rel_rank].active = 0;
            reqs[rel_rank].ts = 0;

            // If we were reserved to that leader and the leader released, cancel reservation.
            // (This is a safety measure for rare races in the simulation.)
            if (reserved && rel_rank == reserved_leader) {
              reserved = 0;
              reserved_leader = -1;
              reserved_leader_ts = -1;
            }
          } break;

          default:
            // Unknown message type; ignore.
            break;
        }
      } while (flag);

      // (3b) After processing messages, recompute whether we can lead a group.
      // We only treat ACKs as complete when we got world-1 ACKs (everyone else).
      int have_all_acks = (ack_count >= (world - 1));

      int group_start = -1; // position of leader in queue (start of block)
      int group_id = -1;    // block index (0..)

      // We can attempt leadership only if:
      // - we are requesting,
      // - we are not already reserved for someone else's group,
      // - we are not already in leader_waiting state.
      if (my_req_active && !reserved && !leader_waiting) {
        if (is_leader_and_ready(rank, P, G, reqs, world, my_req_ts, have_all_acks,
                                &group_start, &group_id, scratch)) {

          // We become leader and start inviting the other members of our block.
          leader_waiting = 1;
          leader_ready_count = 0;

          // Build queue snapshot and invite members in [group_start, group_start+G-1].
          // (n is unused except for potential debugging; we keep it to show intent.)
          int n = build_sorted_queue(reqs, world, scratch);
          (void)n;

          for (int i = group_start; i < group_start + G; i++) {
            int member = scratch[i].rank;

            // Skip sending INVITE to ourselves.
            if (member == rank) continue;

            // Send INVITE to this member.
            Msg inv;
            inv.type = MSG_INVITE;
            inv.ts   = lamport_on_send();
            inv.a    = rank;      // leader_rank
            inv.b    = my_req_ts; // leader_req_ts (identifies the group instance)

            MPI_Send(&inv, 1, MPI_MSG, member, 0, MPI_COMM_WORLD);
          }
        }
      }

      // (3c) If we are leader and collected READY from all other G-1 members,
      // we can send START to the full group (including ourselves).
      if (leader_waiting) {
        if (leader_ready_count >= (G - 1)) {

          // Re-check that the group is still valid (queue may have changed).
          int ok = is_leader_and_ready(rank, P, G, reqs, world, my_req_ts, have_all_acks,
                                       &group_start, &group_id, scratch);

          if (ok) {
            // Send START to each member in our current block.
            int n = build_sorted_queue(reqs, world, scratch);
            (void)n;

            for (int i = group_start; i < group_start + G; i++) {
              int member = scratch[i].rank;

              Msg stmsg;
              stmsg.type = MSG_START;
              stmsg.ts   = lamport_on_send();
              stmsg.a    = rank;     // leader_rank
              stmsg.b    = my_req_ts;// leader_req_ts

              MPI_Send(&stmsg, 1, MPI_MSG, member, 0, MPI_COMM_WORLD);
            }

            // We also consider ourselves started now.
            started = 1;
          } else {
            // If the group is no longer valid, reset leader state and try again later.
            leader_waiting = 0;
            leader_ready_count = 0;
          }
        }
      }

      // (3d) Small sleep to avoid busy-spinning at 100% CPU.
      msleep(2);
    }

    // (4) Tour simulation section:
    // Once started==1, we simulate the tour duration and possible beating/hospital.
    {
      // Print start for debugging/observation.
      // Note: output order between ranks is not deterministic in MPI.
      printf("[rank %d] START trip=%d (lamport=%d)\n", rank, trip, lamport);
      fflush(stdout);

      // Simulate being on the tour.
      msleep(tour_ms);

      // Randomly decide if beaten, then simulate hospital time.
      double r = frand01(&rng);
      if (r < beat_prob) {
        printf("[rank %d] got BEATEN -> hospital %dms (lamport=%d)\n", rank, hospital_ms, lamport);
        fflush(stdout);
        msleep(hospital_ms);
      }

      // Print end for debugging/observation.
      printf("[rank %d] END trip=%d (lamport=%d)\n", rank, trip, lamport);
      fflush(stdout);
    }

    // (5) Release section:
    // After finishing the tour, broadcast REL so everyone removes us from their queue.
    {
      // Remove ourselves locally from the request table.
      reqs[rank].active = 0;
      reqs[rank].ts = 0;
      my_req_active = 0;

      // Build REL message.
      Msg rel;
      rel.type = MSG_REL;
      rel.ts   = lamport_on_send();
      rel.a    = rank; // rank that is releasing
      rel.b    = 0;

      // Broadcast REL to all other processes.
      for (int dst = 0; dst < world; dst++) {
        if (dst == rank) continue;
        MPI_Send(&rel, 1, MPI_MSG, dst, 0, MPI_COMM_WORLD);
      }

      // Small delay so REL messages propagate and states converge faster.
      msleep(10);
    }
  }

  // Cleanup: free the custom MPI datatype for Msg.
  MPI_Type_free(&MPI_MSG);

  // Cleanup: free allocated arrays.
  free(reqs);
  free(scratch);

  // Finalize MPI runtime.
  MPI_Finalize();
  return 0;
}
