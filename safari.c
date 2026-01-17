// safari_mpi.c
// mpicc -O2 -Wall -Wextra safari_mpi.c -o safari
// mpirun -np 20 ./safari 2 4 3 800 1200 0.25
//
// args: P G trips tour_ms hospital_ms beat_prob
// P = liczba przewodnikow (maks. rownoleglych grup)
// G = rozmiar grupy
// trips = ile razy kazdy turysta probuje isc na wycieczke
// tour_ms = czas wycieczki
// hospital_ms = czas w szpitalu po pobiciu
// beat_prob = prawdopodobienstwo pobicia (0..1)

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

typedef enum {
  MSG_REQ = 1,
  MSG_ACK = 2,
  MSG_INVITE = 3,
  MSG_READY = 4,
  MSG_START = 5,
  MSG_REL = 6
} MsgType;

typedef struct {
  int type;      // MsgType
  int ts;        // Lamport timestamp nadawcy
  int a;         // znaczenie zalezne od typu
  int b;         // znaczenie zalezne od typu
} Msg;

// REQ:   a=req_rank, b=req_ts
// ACK:   a=req_rank, b=req_ts
// INVITE:a=leader_rank, b=leader_req_ts
// READY: a=leader_rank, b=leader_req_ts
// START: a=leader_rank, b=leader_req_ts
// REL:   a=rank_to_release, b=0

typedef struct {
  int active;
  int ts;
} ReqInfo;

typedef struct {
  int rank;
  int ts;
} Entry;

static int lamport = 0;

static void lamport_on_recv(int msg_ts) {
  if (msg_ts > lamport) lamport = msg_ts;
  lamport += 1;
}

static int lamport_on_send(void) {
  lamport += 1;
  return lamport;
}

static void msleep(int ms) {
  usleep((useconds_t)ms * 1000u);
}

static int cmp_entry(const void *p1, const void *p2) {
  const Entry *a = (const Entry*)p1;
  const Entry *b = (const Entry*)p2;
  if (a->ts != b->ts) return (a->ts < b->ts) ? -1 : 1;
  return (a->rank < b->rank) ? -1 : (a->rank > b->rank);
}

static int build_sorted_queue(const ReqInfo *reqs, int world_size, Entry *out) {
  int n = 0;
  for (int r = 0; r < world_size; r++) {
    if (reqs[r].active) {
      out[n].rank = r;
      out[n].ts = reqs[r].ts;
      n++;
    }
  }
  qsort(out, (size_t)n, sizeof(Entry), cmp_entry);
  return n;
}

static int find_pos_in_queue(const Entry *q, int n, int my_rank) {
  for (int i = 0; i < n; i++) if (q[i].rank == my_rank) return i;
  return -1;
}

static int is_leader_and_ready(
  int my_rank, int P, int G,
  const ReqInfo *reqs, int world_size,
  int my_req_ts, int have_all_acks,
  int *out_group_start, int *out_group_id, Entry *scratch
) {
  if (!have_all_acks) return 0;
  if (!reqs[my_rank].active || reqs[my_rank].ts != my_req_ts) return 0;

  int n = build_sorted_queue(reqs, world_size, scratch);
  int pos = find_pos_in_queue(scratch, n, my_rank);
  if (pos < 0) return 0;

  if (pos % G != 0) return 0;          // nie jestem liderem bloku
  int grp = pos / G;
  if (grp >= P) return 0;              // brak wolnego przewodnika

  if (pos + (G - 1) >= n) return 0;    // blok niekompletny (za malo chetnych)

  // grupa jest kompletna
  *out_group_start = pos;
  *out_group_id = grp;
  return 1;
}

static int member_belongs_to_leader(
  int my_rank, int P, int G,
  const ReqInfo *reqs, int world_size,
  int leader_rank, int leader_req_ts,
  Entry *scratch
) {
  if (!reqs[my_rank].active) return 0;
  // leader musi byc w kolejce z tym timestampem
  if (!reqs[leader_rank].active || reqs[leader_rank].ts != leader_req_ts) return 0;

  int n = build_sorted_queue(reqs, world_size, scratch);
  int my_pos = find_pos_in_queue(scratch, n, my_rank);
  int leader_pos = find_pos_in_queue(scratch, n, leader_rank);
  if (my_pos < 0 || leader_pos < 0) return 0;

  // leader musi byc poczatkiem bloku
  if (leader_pos % G != 0) return 0;
  int grp = leader_pos / G;
  if (grp >= P) return 0;

  // blok musi byc kompletny
  if (leader_pos + (G - 1) >= n) return 0;

  // ja musze byc w zakresie [leader_pos, leader_pos+G-1]
  return (my_pos >= leader_pos && my_pos <= leader_pos + (G - 1));
}

static double frand01(unsigned int *seed) {
  // prosty LCG + skalowanie
  *seed = (*seed * 1103515245u + 12345u);
  return (double)((*seed / 65536u) % 32768u) / 32768.0;
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank = 0, world = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);

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

  int P = atoi(argv[1]);
  int G = atoi(argv[2]);
  int trips = atoi(argv[3]);
  int tour_ms = atoi(argv[4]);
  int hospital_ms = atoi(argv[5]);
  double beat_prob = atof(argv[6]);

  if (G <= 0 || P <= 0) {
    if (rank == 0) fprintf(stderr, "P and G must be > 0\n");
    MPI_Finalize();
    return 1;
  }
  if (world < 2 * G && rank == 0) {
    fprintf(stderr, "Warning: T (np=%d) < 2*G (%d). Spec says at least 2*G.\n", world, 2*G);
  }

  // MPI datatype for Msg (4 ints)
  MPI_Datatype MPI_MSG;
  MPI_Type_contiguous(4, MPI_INT, &MPI_MSG);
  MPI_Type_commit(&MPI_MSG);

  ReqInfo *reqs = (ReqInfo*)calloc((size_t)world, sizeof(ReqInfo));
  Entry *scratch = (Entry*)malloc((size_t)world * sizeof(Entry));

  // Moje REQ
  int my_req_active = 0;
  int my_req_ts = -1;
  int ack_count = 0;

  // Zaproszenie, ktore zaakceptowalem (rezerwacja)
  int reserved = 0;
  int reserved_leader = -1;
  int reserved_leader_ts = -1;

  // Lider: zebrane READY
  int leader_waiting = 0;
  int leader_ready_count = 0;

  unsigned int rng = (unsigned int)(1234u + 99991u * (unsigned int)rank);

  for (int trip = 0; trip < trips; trip++) {
    // 1) Losowa pauza przed proba (rozsynchronizowanie)
    msleep(50 + (int)(frand01(&rng) * 200.0));

    // 2) Wyslij REQ
    my_req_active = 1;
    ack_count = 0;
    reserved = 0;
    reserved_leader = -1;
    reserved_leader_ts = -1;
    leader_waiting = 0;
    leader_ready_count = 0;

    my_req_ts = lamport_on_send();
    reqs[rank].active = 1;
    reqs[rank].ts = my_req_ts;

    Msg m;
    m.type = MSG_REQ;
    m.ts   = my_req_ts;
    m.a    = rank;
    m.b    = my_req_ts;

    for (int dst = 0; dst < world; dst++) {
      if (dst == rank) continue;
      MPI_Send(&m, 1, MPI_MSG, dst, 0, MPI_COMM_WORLD);
    }

    int started = 0;

    // 3) Petla zdarzen az do START (lub dopoki nie zostane czlonkiem START)
    while (!started) {
      // a) Odbierz wszystkie dostepne wiadomosci (bez blokowania)
      int flag = 0;
      MPI_Status st;

      do {
        MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, &st);
        if (!flag) break;

        Msg rcv;
        MPI_Recv(&rcv, 1, MPI_MSG, st.MPI_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        lamport_on_recv(rcv.ts);

        switch (rcv.type) {
          case MSG_REQ: {
            // Dodaj/zaktualizuj wpis w kolejce
            int req_rank = rcv.a;
            int req_ts   = rcv.b;
            // zachowaj najnowszy aktywny request (w praktyce tu zakladamy max jeden na proces)
            reqs[req_rank].active = 1;
            reqs[req_rank].ts = req_ts;

            // Odeslij ACK dla konkretnego (rank, ts)
            Msg ack;
            ack.type = MSG_ACK;
            ack.ts   = lamport_on_send();
            ack.a    = req_rank;
            ack.b    = req_ts;
            MPI_Send(&ack, 1, MPI_MSG, st.MPI_SOURCE, 0, MPI_COMM_WORLD);
          } break;

          case MSG_ACK: {
            // ACK dotyczy (a=req_rank, b=req_ts)
            if (my_req_active && rcv.a == rank && rcv.b == my_req_ts) {
              ack_count++;
            }
          } break;

          case MSG_INVITE: {
            int leader = rcv.a;
            int leader_ts = rcv.b;

            // Jesli jestem w REQUESTING i faktycznie naleze do bloku lidera -> READY
            if (my_req_active && !reserved) {
              if (member_belongs_to_leader(rank, P, G, reqs, world, leader, leader_ts, scratch)) {
                reserved = 1;
                reserved_leader = leader;
                reserved_leader_ts = leader_ts;

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
            // tylko lider zbiera READY
            if (leader_waiting && my_req_active) {
              if (rcv.a == rank && rcv.b == my_req_ts) {
                leader_ready_count++;
              }
            }
          } break;

          case MSG_START: {
            int leader = rcv.a;
            int leader_ts = rcv.b;

            // START mnie dotyczy, jesli naleze do lidera
            if (my_req_active &&
                member_belongs_to_leader(rank, P, G, reqs, world, leader, leader_ts, scratch)) {
              started = 1;
            }
          } break;

          case MSG_REL: {
            int rel_rank = rcv.a;
            reqs[rel_rank].active = 0;
            reqs[rel_rank].ts = 0;
            // jesli ja bylem zarezerwowany do lidera, a lider znika -> anuluj rezerwacje
            if (reserved && rel_rank == reserved_leader) {
              reserved = 0;
              reserved_leader = -1;
              reserved_leader_ts = -1;
            }
          } break;

          default:
            break;
        }
      } while (flag);

      // b) Logika lidera: czy moge zostac liderem i wyslac INVITE?
      int have_all_acks = (ack_count >= (world - 1));
      int group_start = -1, group_id = -1;

      if (my_req_active && !reserved && !leader_waiting) {
        if (is_leader_and_ready(rank, P, G, reqs, world, my_req_ts, have_all_acks,
                                &group_start, &group_id, scratch)) {
          // Zostaje liderem i zapraszam pozostalych G-1
          leader_waiting = 1;
          leader_ready_count = 0;

          // Zbuduj kolejke i wez czlonkow bloku [group_start .. group_start+G-1]
          int n = build_sorted_queue(reqs, world, scratch);
          (void)n;

          for (int i = group_start; i < group_start + G; i++) {
            int member = scratch[i].rank;
            if (member == rank) continue;

            Msg inv;
            inv.type = MSG_INVITE;
            inv.ts   = lamport_on_send();
            inv.a    = rank;      // leader_rank
            inv.b    = my_req_ts; // leader_req_ts
            MPI_Send(&inv, 1, MPI_MSG, member, 0, MPI_COMM_WORLD);
          }
        }
      }

      // c) Start jesli lider ma READY od G-1
      if (leader_waiting) {
        if (leader_ready_count >= (G - 1)) {
          // Upewnij sie, ze grupa nadal jest poprawna (nie rozjechala sie kolejka)
          int ok = is_leader_and_ready(rank, P, G, reqs, world, my_req_ts, have_all_acks,
                                       &group_start, &group_id, scratch);
          if (ok) {
            int n = build_sorted_queue(reqs, world, scratch);
            (void)n;

            for (int i = group_start; i < group_start + G; i++) {
              int member = scratch[i].rank;
              Msg stmsg;
              stmsg.type = MSG_START;
              stmsg.ts   = lamport_on_send();
              stmsg.a    = rank;
              stmsg.b    = my_req_ts;
              MPI_Send(&stmsg, 1, MPI_MSG, member, 0, MPI_COMM_WORLD);
            }
            started = 1;
          } else {
            // grupa sie zmienila/rozpadla -> zresetuj i poczekaj na nowe warunki
            leader_waiting = 0;
            leader_ready_count = 0;
          }
        }
      }

      // d) male uspienie, zeby nie mielic CPU w petli
      msleep(2);
    }

    // 4) Wycieczka
    {
      // (opcjonalnie) wypisz start w sposób czytelniejszy
      // Uwaga: kolejność printów jest losowa w MPI
      printf("[rank %d] START trip=%d (lamport=%d)\n", rank, trip, lamport);
      fflush(stdout);

      msleep(tour_ms);

      double r = frand01(&rng);
      if (r < beat_prob) {
        printf("[rank %d] got BEATEN -> hospital %dms\n", rank, hospital_ms);
        fflush(stdout);
        msleep(hospital_ms);
      }

      printf("[rank %d] END trip=%d\n", rank, trip);
      fflush(stdout);
    }

    // 5) Zwolnij (REL) i wyczysc lokalny REQ
    {
      reqs[rank].active = 0;
      reqs[rank].ts = 0;
      my_req_active = 0;

      Msg rel;
      rel.type = MSG_REL;
      rel.ts   = lamport_on_send();
      rel.a    = rank;
      rel.b    = 0;

      for (int dst = 0; dst < world; dst++) {
        if (dst == rank) continue;
        MPI_Send(&rel, 1, MPI_MSG, dst, 0, MPI_COMM_WORLD);
      }

      // chwila na “rozlanie się” REL po systemie
      msleep(10);
    }
  }

  // Sprzatnij typ MPI
  MPI_Type_free(&MPI_MSG);

  free(reqs);
  free(scratch);

  MPI_Finalize();
  return 0;
}
