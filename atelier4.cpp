// ============================================================================
//  ATELIER 4  -  chess engine (C++17, single file, UCI)
//  bitboards + magic bitboards, legal move generation (perft-verified),
//  PVS/aspiration/TT/NMP/ProbCut/singular/LMR/SEE/history+correction-history,
//  tapered hand-crafted evaluation, KPK bitbase, KXK mop-up, lazy-SMP threads,
//  time management, MultiPV, built-in self-test + match helper commands.
// ============================================================================
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <chrono>
#include <atomic>
#include <thread>
#include <mutex>
#include <memory>
#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

typedef uint64_t U64;
typedef uint16_t Move;
enum Color { WHITE, BLACK };
enum PType { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, NO_TYPE };
enum Piece { W_PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING, B_PAWN, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING, NO_PIECE };
enum Sq { A1, B1, C1, D1, E1, F1, G1, H1, A2 = 8, A3 = 16, A4 = 24, A5 = 32, A6 = 40, A7 = 48, A8 = 56, H8 = 63, NO_SQ = 64 };
static const Move MOVE_NONE = 0;
enum MoveFlag { QUIET = 0, DPUSH = 1, OO = 2, OOO = 3, CAPTURE = 4, EP = 5, PROMO_N = 8, PROMO_B = 9, PROMO_R = 10, PROMO_Q = 11,
                PCAP_N = 12, PCAP_B = 13, PCAP_R = 14, PCAP_Q = 15 };
inline Move mk(int f, int t, int fl) { return (Move)(f | (t << 6) | (fl << 12)); }
inline int mFrom(Move m) { return m & 63; }
inline int mTo(Move m) { return (m >> 6) & 63; }
inline int mFlag(Move m) { return m >> 12; }
inline bool isCap(Move m) { return mFlag(m) & 4; }
inline bool isPromo(Move m) { return mFlag(m) & 8; }
inline int promoType(Move m) { return (mFlag(m) & 3) + KNIGHT; }
inline bool isNoisy(Move m) { return mFlag(m) & 12; }  // capture or promotion
inline int fileOf(int s) { return s & 7; }
inline int rankOf(int s) { return s >> 3; }
inline int typeOf(int p) { return p >= 6 ? p - 6 : p; }
inline int colorOf(int p) { return p >= 6; }
inline int mkPiece(int c, int t) { return c * 6 + t; }
inline int relRank(int c, int s) { return c == WHITE ? rankOf(s) : 7 - rankOf(s); }
inline int relSq(int c, int s) { return c == WHITE ? s : s ^ 56; }

#define popcnt(b) __builtin_popcountll(b)
#define lsb(b) __builtin_ctzll(b)
inline int popLsb(U64& b) { int s = lsb(b); b &= b - 1; return s; }
inline U64 bit(int s) { return 1ULL << s; }
inline bool more1(U64 b) { return b & (b - 1); }

static const U64 FILE_A = 0x0101010101010101ULL, RANK_1 = 0xFFULL;
static const U64 LIGHT_SQ = 0x55AA55AA55AA55AAULL, DARK_SQ = ~LIGHT_SQ;
static U64 FileBB[8], RankBB[8], AdjFiles[8], PawnAtt[2][64], KnightAtt[64], KingAtt[64];
static U64 BetweenBB[64][64], LineBB[64][64], PassedMask[2][64], ForwardMask[2][64], KingZone[2][64];
static int Dist[64][64];

struct Magic { U64 mask, magic; U64* att; int shift; };
static Magic RMagic[64], BMagic[64];
static U64 RTable[102400], BTable[5248];

static U64 slowAtt(int sq, U64 occ, const int* dirs) {
    U64 r = 0;
    for (int i = 0; i < 4; i++) {
        int df = dirs[2 * i], dr = dirs[2 * i + 1], f = fileOf(sq) + df, rk = rankOf(sq) + dr;
        while (f >= 0 && f < 8 && rk >= 0 && rk < 8) { int s = rk * 8 + f; r |= bit(s); if (occ & bit(s)) break; f += df; rk += dr; }
    }
    return r;
}
static const int RDIR[8] = {1, 0, -1, 0, 0, 1, 0, -1}, BDIR[8] = {1, 1, 1, -1, -1, 1, -1, -1};
static U64 rngState = 0x9E3779B97F4A7C15ULL;
static U64 rnd64() { rngState ^= rngState >> 12; rngState ^= rngState << 25; rngState ^= rngState >> 27; return rngState * 2685821657736338717ULL; }
static U64 sparseRnd() { return rnd64() & rnd64() & rnd64(); }

static void initMagics(bool rook, Magic* M, U64* table) {
    const int* dirs = rook ? RDIR : BDIR;
    U64* ptr = table;
    static U64 occs[4096], refs[4096];
    for (int sq = 0; sq < 64; sq++) {
        U64 edges = ((RANK_1 | (RANK_1 << 56)) & ~RankBB[rankOf(sq)]) | ((FILE_A | (FILE_A << 7)) & ~FileBB[fileOf(sq)]);
        U64 mask = slowAtt(sq, 0, dirs) & ~edges;
        int bits = popcnt(mask), size = 1 << bits;
        M[sq].mask = mask; M[sq].shift = 64 - bits; M[sq].att = ptr;
        U64 b = 0; int n = 0;
        do { occs[n] = b; refs[n] = slowAtt(sq, b, dirs); n++; b = (b - mask) & mask; } while (b);
        static U64 used[4096]; static int epoch[4096]; static int ep = 0;
        for (;;) {
            U64 magic = sparseRnd();
            if (popcnt((mask * magic) >> 56) < 6) continue;
            ep++; bool ok = true;
            for (int i = 0; i < n && ok; i++) {
                int idx = (int)((occs[i] * magic) >> M[sq].shift);
                if (epoch[idx] != ep) { epoch[idx] = ep; used[idx] = refs[i]; }
                else if (used[idx] != refs[i]) ok = false;
            }
            if (ok) { M[sq].magic = magic; for (int i = 0; i < n; i++) ptr[(int)((occs[i] * magic) >> M[sq].shift)] = refs[i]; break; }
        }
        ptr += size;
    }
}
inline U64 rookAtt(int s, U64 occ) { const Magic& m = RMagic[s]; return m.att[((occ & m.mask) * m.magic) >> m.shift]; }
inline U64 bishopAtt(int s, U64 occ) { const Magic& m = BMagic[s]; return m.att[((occ & m.mask) * m.magic) >> m.shift]; }
inline U64 queenAtt(int s, U64 occ) { return rookAtt(s, occ) | bishopAtt(s, occ); }
inline U64 pawnAttsBB(int c, U64 pawns) {
    return c == WHITE ? (((pawns & ~FILE_A) << 7) | ((pawns & ~(FILE_A << 7)) << 9)) : (((pawns & ~FILE_A) >> 9) | ((pawns & ~(FILE_A << 7)) >> 7));
}
inline U64 shiftUp(int c, U64 b) { return c == WHITE ? b << 8 : b >> 8; }

static U64 ZobPc[12][64], ZobCastle[16], ZobEp[8], ZobSide;
static int CastleMask[64];

static void initTables() {
    for (int i = 0; i < 8; i++) { FileBB[i] = FILE_A << i; RankBB[i] = RANK_1 << (8 * i); }
    for (int i = 0; i < 8; i++) AdjFiles[i] = (i > 0 ? FileBB[i - 1] : 0) | (i < 7 ? FileBB[i + 1] : 0);
    for (int s = 0; s < 64; s++) {
        int f = fileOf(s), r = rankOf(s);
        PawnAtt[WHITE][s] = pawnAttsBB(WHITE, bit(s)); PawnAtt[BLACK][s] = pawnAttsBB(BLACK, bit(s));
        KnightAtt[s] = KingAtt[s] = 0;
        const int kn[8][2] = {{1, 2}, {2, 1}, {-1, 2}, {-2, 1}, {1, -2}, {2, -1}, {-1, -2}, {-2, -1}};
        for (auto& d : kn) { int nf = f + d[0], nr = r + d[1]; if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) KnightAtt[s] |= bit(nr * 8 + nf); }
        for (int df = -1; df <= 1; df++) for (int dr = -1; dr <= 1; dr++) if (df || dr) {
            int nf = f + df, nr = r + dr; if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) KingAtt[s] |= bit(nr * 8 + nf); }
        for (int t = 0; t < 64; t++) Dist[s][t] = std::max(std::abs(f - fileOf(t)), std::abs(r - rankOf(t)));
    }
    initMagics(true, RMagic, RTable); initMagics(false, BMagic, BTable);
    for (int a = 0; a < 64; a++) for (int b = 0; b < 64; b++) {
        BetweenBB[a][b] = LineBB[a][b] = 0;
        if (a == b) continue;
        if (rookAtt(a, 0) & bit(b)) { LineBB[a][b] = (rookAtt(a, 0) & rookAtt(b, 0)) | bit(a) | bit(b); BetweenBB[a][b] = rookAtt(a, bit(b)) & rookAtt(b, bit(a)); }
        else if (bishopAtt(a, 0) & bit(b)) { LineBB[a][b] = (bishopAtt(a, 0) & bishopAtt(b, 0)) | bit(a) | bit(b); BetweenBB[a][b] = bishopAtt(a, bit(b)) & bishopAtt(b, bit(a)); }
    }
    for (int c = 0; c < 2; c++) for (int s = 0; s < 64; s++) {
        U64 fw = 0; int r = rankOf(s);
        for (int rr = c == WHITE ? r + 1 : r - 1; rr >= 0 && rr < 8; rr += (c == WHITE ? 1 : -1)) fw |= RankBB[rr];
        ForwardMask[c][s] = fw & FileBB[fileOf(s)];
        PassedMask[c][s] = fw & (FileBB[fileOf(s)] | AdjFiles[fileOf(s)]);
        U64 z = KingAtt[s] | bit(s);
        U64 up = c == WHITE ? (z << 8) : (z >> 8);
        KingZone[c][s] = z | up;
    }
    U64 seed = 0x1234567ULL;
    auto r = [&]() { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; return seed * 0x2545F4914F6CDD1DULL; };
    for (int p = 0; p < 12; p++) for (int s = 0; s < 64; s++) ZobPc[p][s] = r();
    for (int i = 0; i < 16; i++) ZobCastle[i] = r();
    for (int i = 0; i < 8; i++) ZobEp[i] = r();
    ZobSide = r();
    for (int s = 0; s < 64; s++) CastleMask[s] = 15;
    CastleMask[A1] &= ~2; CastleMask[H1] &= ~1; CastleMask[E1] &= ~3;
    CastleMask[A8] &= ~8; CastleMask[H8] &= ~4; CastleMask[60] &= ~12;
}
// ------------------------------- Position -----------------------------------
static int PSQ[12][64];  // packed (mg,eg), filled by initEval()
static const int PieceVal[7] = {100, 320, 330, 500, 900, 20000, 0};
inline int S(int mg, int eg) { return (int)((unsigned)eg << 16) + mg; }
inline int mgOf(int s) { return (int16_t)((unsigned)s & 0xffff); }
inline int egOf(int s) { return (int16_t)(((unsigned)s + 0x8000) >> 16); }

struct StateInfo {
    U64 key, pawnKey, npKey[2], checkers, pinned;
    int castling, ep, halfmove, captured, psq;
    Move move;
};
#ifndef STACK_SZ_OVERRIDE
static const int STACK_SZ = 2600;
#else
static const int STACK_SZ = STACK_SZ_OVERRIDE;
#endif
static const char* PCH = "PNBRQKpnbrqk";

struct Position {
    U64 byType[6], byColor[2], occ;
    int board[64];
    int side, fullmove, sp;
    StateInfo st[STACK_SZ];

    StateInfo& S0() { return st[sp]; }
    const StateInfo& S0() const { return st[sp]; }
    U64 pieces(int c, int t) const { return byType[t] & byColor[c]; }
    int ksq(int c) const { return lsb(byType[KING] & byColor[c]); }
    U64 key() const { return st[sp].key; }
    U64 checkers() const { return st[sp].checkers; }
    bool inCheck() const { return st[sp].checkers != 0; }

    void clear() {
        memset(byType, 0, sizeof byType); memset(byColor, 0, sizeof byColor); occ = 0;
        for (int i = 0; i < 64; i++) board[i] = NO_PIECE;
        side = WHITE; fullmove = 1; sp = 0;
        StateInfo& s = st[0]; memset(&s, 0, sizeof s); s.ep = NO_SQ;
    }
    void put(int p, int s) { board[s] = p; U64 b = bit(s); byType[typeOf(p)] |= b; byColor[colorOf(p)] |= b; occ |= b; }
    void rem(int s) { int p = board[s]; U64 b = bit(s); byType[typeOf(p)] ^= b; byColor[colorOf(p)] ^= b; occ ^= b; board[s] = NO_PIECE; }

    U64 attackersTo(int s, U64 o) const {
        return (PawnAtt[BLACK][s] & pieces(WHITE, PAWN)) | (PawnAtt[WHITE][s] & pieces(BLACK, PAWN)) | (KnightAtt[s] & byType[KNIGHT]) |
               (KingAtt[s] & byType[KING]) | (rookAtt(s, o) & (byType[ROOK] | byType[QUEEN])) | (bishopAtt(s, o) & (byType[BISHOP] | byType[QUEEN]));
    }
    bool attacked(int s, int by, U64 o) const { return (attackersTo(s, o) & byColor[by]) != 0; }

    void computeCheckInfo() {
        StateInfo& s = st[sp]; int us = side, them = us ^ 1, k = ksq(us);
        s.checkers = attackersTo(k, occ) & byColor[them];
        s.pinned = 0;
        U64 snip = (rookAtt(k, 0) & (pieces(them, ROOK) | pieces(them, QUEEN))) | (bishopAtt(k, 0) & (pieces(them, BISHOP) | pieces(them, QUEEN)));
        while (snip) {
            int sn = popLsb(snip); U64 b = BetweenBB[k][sn] & occ;
            if (b && !more1(b)) s.pinned |= b & byColor[us];
        }
    }
    void computeKeys() {
        StateInfo& s = st[sp]; s.key = s.pawnKey = 0; s.npKey[0] = s.npKey[1] = 0; s.psq = 0;
        for (int sq = 0; sq < 64; sq++) if (board[sq] != NO_PIECE) {
            int p = board[sq]; s.key ^= ZobPc[p][sq]; s.psq += PSQ[p][sq];
            if (typeOf(p) == PAWN) s.pawnKey ^= ZobPc[p][sq]; else s.npKey[colorOf(p)] ^= ZobPc[p][sq];
        }
        s.key ^= ZobCastle[s.castling]; if (s.ep != NO_SQ) s.key ^= ZobEp[fileOf(s.ep)]; if (side == BLACK) s.key ^= ZobSide;
    }

    bool setFen(const std::string& fen) {
        clear(); std::istringstream ss(fen); std::string bd, sd, cs, eps; int hm = 0, fm = 1;
        if (!(ss >> bd >> sd)) return false;
        ss >> cs >> eps >> hm >> fm;
        int r = 7, f = 0, wk = 0, bk = 0;
        for (char c : bd) {
            if (c == '/') { r--; f = 0; if (r < 0) return false; }
            else if (c >= '1' && c <= '8') f += c - '0';
            else { const char* p = strchr(PCH, c); if (!p || f > 7) return false; int pc = (int)(p - PCH); if (pc == W_KING) wk++; if (pc == B_KING) bk++; put(pc, r * 8 + f); f++; }
        }
        if (wk != 1 || bk != 1 || r != 0) return false;
        side = sd == "b" ? BLACK : WHITE;
        StateInfo& s = st[0]; s.castling = 0;
        for (char c : cs) {
            if (c == 'K' && board[E1] == W_KING && board[H1] == W_ROOK) s.castling |= 1;
            if (c == 'Q' && board[E1] == W_KING && board[A1] == W_ROOK) s.castling |= 2;
            if (c == 'k' && board[60] == B_KING && board[H8] == B_ROOK) s.castling |= 4;
            if (c == 'q' && board[60] == B_KING && board[A8] == B_ROOK) s.castling |= 8;
        }
        s.ep = NO_SQ;
        if (eps.size() == 2 && eps[0] >= 'a' && eps[0] <= 'h' && (eps[1] == '3' || eps[1] == '6')) {
            int e = (eps[1] - '1') * 8 + (eps[0] - 'a'); int us = side;
            if (PawnAtt[us ^ 1][e] & pieces(us, PAWN)) s.ep = e;  // only keep if capturable
        }
        s.halfmove = hm; fullmove = std::max(1, fm);
        // opponent must not be in check (illegal position)
        if (attacked(ksq(side ^ 1), side, occ)) return false;
        computeKeys(); computeCheckInfo();
        return true;
    }
    std::string fen() const {
        std::string o;
        for (int r = 7; r >= 0; r--) { int e = 0; for (int f = 0; f < 8; f++) { int p = board[r * 8 + f]; if (p == NO_PIECE) e++; else { if (e) o += char('0' + e); e = 0; o += PCH[p]; } } if (e) o += char('0' + e); if (r) o += '/'; }
        o += side == WHITE ? " w " : " b "; int c = st[sp].castling;
        if (!c) o += "-"; else { if (c & 1) o += 'K'; if (c & 2) o += 'Q'; if (c & 4) o += 'k'; if (c & 8) o += 'q'; }
        o += " "; int e = st[sp].ep; if (e == NO_SQ) o += "-"; else { o += char('a' + fileOf(e)); o += char('1' + rankOf(e)); }
        o += " " + std::to_string(st[sp].halfmove) + " " + std::to_string(fullmove); return o;
    }

    void doMove(Move m) {
        StateInfo& o = st[sp]; StateInfo& n = st[sp + 1]; n = o; sp++;
        n.move = m; n.captured = NO_PIECE; n.halfmove++;
        int us = side, from = mFrom(m), to = mTo(m), fl = mFlag(m), pc = board[from];
        if (n.ep != NO_SQ) { n.key ^= ZobEp[fileOf(n.ep)]; n.ep = NO_SQ; }
        int nc = n.castling & CastleMask[from] & CastleMask[to];
        if (nc != n.castling) { n.key ^= ZobCastle[n.castling] ^ ZobCastle[nc]; n.castling = nc; }
        auto nonpawn = [&](int p, int sq) { if (typeOf(p) != PAWN) n.npKey[colorOf(p)] ^= ZobPc[p][sq]; };
        if (fl == EP) {
            int cs = to + (us == WHITE ? -8 : 8), cp = board[cs]; rem(cs); n.key ^= ZobPc[cp][cs]; n.pawnKey ^= ZobPc[cp][cs]; n.psq -= PSQ[cp][cs]; n.captured = cp; n.halfmove = 0;
        } else if (fl & 4) {
            int cp = board[to]; rem(to); n.key ^= ZobPc[cp][to]; n.psq -= PSQ[cp][to]; n.captured = cp; n.halfmove = 0;
            if (typeOf(cp) == PAWN) n.pawnKey ^= ZobPc[cp][to]; else n.npKey[colorOf(cp)] ^= ZobPc[cp][to];
        }
        rem(from); n.key ^= ZobPc[pc][from]; n.psq -= PSQ[pc][from];
        if (typeOf(pc) == PAWN) n.pawnKey ^= ZobPc[pc][from]; else nonpawn(pc, from);
        int np = pc;
        if (fl & 8) np = mkPiece(us, promoType(m));
        put(np, to); n.key ^= ZobPc[np][to]; n.psq += PSQ[np][to];
        if (typeOf(np) == PAWN) n.pawnKey ^= ZobPc[np][to]; else nonpawn(np, to);
        if (typeOf(pc) == PAWN) {
            n.halfmove = 0;
            if (fl == DPUSH) { int e = (from + to) / 2; if (PawnAtt[us][e] & pieces(us ^ 1, PAWN)) { n.ep = e; n.key ^= ZobEp[fileOf(e)]; } }
        }
        if (fl == OO || fl == OOO) {
            int rf = fl == OO ? to + 1 : to - 2, rt = fl == OO ? to - 1 : to + 1, rp = board[rf];
            rem(rf); put(rp, rt); n.key ^= ZobPc[rp][rf] ^ ZobPc[rp][rt]; n.npKey[us] ^= ZobPc[rp][rf] ^ ZobPc[rp][rt]; n.psq += PSQ[rp][rt] - PSQ[rp][rf];
        }
        side ^= 1; n.key ^= ZobSide; if (us == BLACK) fullmove++;
        computeCheckInfo();
    }
    void undoMove() {
        StateInfo& n = st[sp]; Move m = n.move; side ^= 1; int us = side, from = mFrom(m), to = mTo(m), fl = mFlag(m);
        if (us == BLACK) fullmove--;
        if (fl == OO || fl == OOO) { int rf = fl == OO ? to + 1 : to - 2, rt = fl == OO ? to - 1 : to + 1, rp = board[rt]; rem(rt); put(rp, rf); }
        int pc = board[to]; rem(to);
        if (fl & 8) pc = mkPiece(us, PAWN);
        put(pc, from);
        if (fl == EP) put(n.captured, to + (us == WHITE ? -8 : 8));
        else if (fl & 4) put(n.captured, to);
        sp--;
    }
    void doNull() { StateInfo& o = st[sp]; StateInfo& n = st[sp + 1]; n = o; sp++; n.move = MOVE_NONE; n.captured = NO_PIECE;
        if (n.ep != NO_SQ) { n.key ^= ZobEp[fileOf(n.ep)]; n.ep = NO_SQ; }
        n.halfmove++; side ^= 1; n.key ^= ZobSide; computeCheckInfo(); }
    void undoNull() { side ^= 1; sp--; }

    // repetition: inside the search tree (ply) one earlier occurrence is enough; before the root two are needed
    bool isRepetition(int ply) const {
        const StateInfo& s = st[sp]; int hm = std::min(s.halfmove, sp), cnt = 0;
        for (int i = 4; i <= hm; i += 2) if (st[sp - i].key == s.key) { if (i <= ply) return true; if (++cnt >= 2) return true; }
        return false;
    }
    bool insufficient() const {
        if (byType[PAWN] | byType[ROOK] | byType[QUEEN]) return false;
        int n = popcnt(byType[KNIGHT]), b = popcnt(byType[BISHOP]);
        if (n + b <= 1) return true;
        if (n == 0 && b >= 2) { return !((byType[BISHOP] & LIGHT_SQ) && (byType[BISHOP] & DARK_SQ)); }  // all bishops on one colour
        return false;
    }
    bool isLegal(Move m) const {
        const StateInfo& s = st[sp]; int us = side, them = us ^ 1, k = ksq(us), from = mFrom(m), to = mTo(m), fl = mFlag(m);
        if (fl == EP) {
            int cs = to + (us == WHITE ? -8 : 8); U64 o = (occ ^ bit(from) ^ bit(cs)) | bit(to);
            return !((rookAtt(k, o) & (pieces(them, ROOK) | pieces(them, QUEEN))) | (bishopAtt(k, o) & (pieces(them, BISHOP) | pieces(them, QUEEN))) |
                     (KnightAtt[k] & pieces(them, KNIGHT)) | (PawnAtt[us][k] & pieces(them, PAWN) & ~bit(cs)));
        }
        if (from == k) { if (fl == OO || fl == OOO) return true; return !attacked(to, them, occ ^ bit(from)); }
        if (s.checkers) {
            if (more1(s.checkers)) return false;
            int c = lsb(s.checkers); if (!(bit(to) & (BetweenBB[k][c] | bit(c)))) return false;
        }
        if (s.pinned & bit(from)) return (LineBB[from][to] & bit(k)) != 0;
        return true;
    }
    bool isPseudoLegal(Move m) const {
        if (m == MOVE_NONE) return false;
        int us = side, from = mFrom(m), to = mTo(m), fl = mFlag(m), pc = board[from];
        if (pc == NO_PIECE || colorOf(pc) != us || from == to) return false;
        int t = typeOf(pc), tp = board[to];
        if (tp != NO_PIECE && colorOf(tp) == us) return false;
        if (typeOf(tp) == KING && tp != NO_PIECE) return false;
        const StateInfo& s = st[sp];
        if (fl == OO || fl == OOO) {
            if (t != KING || s.checkers) return false;
            int base = us == WHITE ? 0 : 56; if (from != base + 4) return false;
            if (fl == OO) { if (!(s.castling & (us == WHITE ? 1 : 4)) || to != base + 6 || (occ & (bit(base + 5) | bit(base + 6))) || board[base + 7] != mkPiece(us, ROOK)) return false;
                return !attacked(base + 5, us ^ 1, occ) && !attacked(base + 6, us ^ 1, occ) && isLegal(m); }
            if (!(s.castling & (us == WHITE ? 2 : 8)) || to != base + 2 || (occ & (bit(base + 1) | bit(base + 2) | bit(base + 3))) || board[base] != mkPiece(us, ROOK)) return false;
            return !attacked(base + 3, us ^ 1, occ) && !attacked(base + 2, us ^ 1, occ) && isLegal(m);
        }
        if (t == PAWN) {
            int dir = us == WHITE ? 8 : -8, rr = relRank(us, to);
            bool promoRank = rr == 7; if (promoRank != (bool)(fl & 8)) return false;
            if (fl == EP) { if (to != s.ep || tp != NO_PIECE || !(PawnAtt[us][from] & bit(to))) return false; }
            else if (fl & 4) { if (tp == NO_PIECE || !(PawnAtt[us][from] & bit(to))) return false; }
            else if (fl == DPUSH) { if (relRank(us, from) != 1 || to != from + 2 * dir || board[from + dir] != NO_PIECE || tp != NO_PIECE) return false; }
            else if (fl == QUIET || (fl & 8)) { if (to != from + dir || tp != NO_PIECE) return false; if ((fl & 4)) return false; }
            else return false;
            return isLegal(m);
        }
        if (fl != QUIET && fl != CAPTURE) return false;
        if ((fl == CAPTURE) != (tp != NO_PIECE)) return false;
        U64 a = t == KNIGHT ? KnightAtt[from] : t == BISHOP ? bishopAtt(from, occ) : t == ROOK ? rookAtt(from, occ) : t == QUEEN ? queenAtt(from, occ) : KingAtt[from];
        if (!(a & bit(to))) return false;
        return isLegal(m);
    }
    bool givesCheck(Move m) const {
        int us = side, ek = ksq(us ^ 1), from = mFrom(m), to = mTo(m), fl = mFlag(m), pc = board[from], t = typeOf(pc);
        U64 o = (occ ^ bit(from)) | bit(to);
        if (fl == EP) o ^= bit(to + (us == WHITE ? -8 : 8));
        int nt = (fl & 8) ? promoType(m) : t;
        U64 ra, ba;
        switch (nt) {
            case PAWN: if (PawnAtt[us][to] & bit(ek)) return true; break;
            case KNIGHT: if (KnightAtt[to] & bit(ek)) return true; break;
            case BISHOP: if (bishopAtt(to, o) & bit(ek)) return true; break;
            case ROOK: if (rookAtt(to, o) & bit(ek)) return true; break;
            case QUEEN: if (queenAtt(to, o) & bit(ek)) return true; break;
        }
        if (fl == OO || fl == OOO) {
            int rt = fl == OO ? to - 1 : to + 1, rf = fl == OO ? to + 1 : to - 2; o = (o ^ bit(rf)) | bit(rt);
            if (rookAtt(rt, o) & bit(ek)) return true;
        }
        ra = rookAtt(ek, o) & (pieces(us, ROOK) | pieces(us, QUEEN)) & ~bit(from);
        ba = bishopAtt(ek, o) & (pieces(us, BISHOP) | pieces(us, QUEEN)) & ~bit(from);
        return (ra | ba) != 0;
    }
    bool seeGE(Move m, int th) const {
        int fl = mFlag(m); if (fl == OO || fl == OOO) return 0 >= th;
        int from = mFrom(m), to = mTo(m);
        int swap = (fl == EP ? PieceVal[PAWN] : (board[to] == NO_PIECE ? 0 : PieceVal[typeOf(board[to])])) - th;
        if (fl & 8) swap += PieceVal[promoType(m)] - PieceVal[PAWN];
        if (swap < 0) return false;
        int mv = (fl & 8) ? PieceVal[promoType(m)] : PieceVal[typeOf(board[from])];
        swap = mv - swap; if (swap <= 0) return true;
        U64 o = occ ^ bit(from) ^ bit(to); if (fl == EP) o ^= bit(to + (side == WHITE ? -8 : 8));
        int stm = side; U64 att = attackersTo(to, o); int res = 1;
        while (true) {
            stm ^= 1; att &= o; U64 sa = att & byColor[stm]; if (!sa) break;
            res ^= 1; U64 bb;
            if ((bb = sa & byType[PAWN])) { if ((swap = PieceVal[PAWN] - swap) < res) break; o ^= bit(lsb(bb)); att |= bishopAtt(to, o) & (byType[BISHOP] | byType[QUEEN]); }
            else if ((bb = sa & byType[KNIGHT])) { if ((swap = PieceVal[KNIGHT] - swap) < res) break; o ^= bit(lsb(bb)); }
            else if ((bb = sa & byType[BISHOP])) { if ((swap = PieceVal[BISHOP] - swap) < res) break; o ^= bit(lsb(bb)); att |= bishopAtt(to, o) & (byType[BISHOP] | byType[QUEEN]); }
            else if ((bb = sa & byType[ROOK])) { if ((swap = PieceVal[ROOK] - swap) < res) break; o ^= bit(lsb(bb)); att |= rookAtt(to, o) & (byType[ROOK] | byType[QUEEN]); }
            else if ((bb = sa & byType[QUEEN])) { if ((swap = PieceVal[QUEEN] - swap) < res) break; o ^= bit(lsb(bb));
                att |= (bishopAtt(to, o) & (byType[BISHOP] | byType[QUEEN])) | (rookAtt(to, o) & (byType[ROOK] | byType[QUEEN])); }
            else return (att & ~byColor[stm]) ? res ^ 1 : res;
        }
        return res;
    }
};

// ------------------------------ Move generation -----------------------------
enum GenMode { G_NOISY, G_QUIET, G_EVASION };
static inline Move* addPawnMoves(Move* l, int from, int to, int fl_base, bool cap, int mode) {
    // promotion expansion (to on last rank)
    if (cap) { *l++ = mk(from, to, PCAP_Q); *l++ = mk(from, to, PCAP_N); *l++ = mk(from, to, PCAP_R); *l++ = mk(from, to, PCAP_B); }
    else {
        if (mode != G_QUIET) *l++ = mk(from, to, PROMO_Q);
        if (mode != G_NOISY) { *l++ = mk(from, to, PROMO_N); *l++ = mk(from, to, PROMO_R); *l++ = mk(from, to, PROMO_B); }
    }
    return l;
}
// Generates pseudo-legal moves; caller filters with isLegal().
static Move* genMoves(const Position& p, Move* l, int mode) {
    int us = p.side, them = us ^ 1, k = p.ksq(us); U64 occ = p.occ, ours = p.byColor[us], theirs = p.byColor[them];
    U64 target; const StateInfo& s = p.st[p.sp];
    bool evas = mode == G_EVASION;
    if (evas) {
        if (more1(s.checkers)) target = 0;
        else { int c = lsb(s.checkers); target = BetweenBB[k][c] | bit(c); }
    } else target = mode == G_NOISY ? theirs : ~occ;
    // king
    {
        U64 km = evas ? (KingAtt[k] & ~ours) : (mode == G_NOISY ? KingAtt[k] & theirs : KingAtt[k] & ~occ);
        while (km) { int to = popLsb(km); *l++ = mk(k, to, (p.board[to] != NO_PIECE) ? CAPTURE : QUIET); }
    }
    if (!(evas && more1(s.checkers))) {
        for (int t = KNIGHT; t <= QUEEN; t++) {
            U64 pc = p.pieces(us, t);
            while (pc) {
                int from = popLsb(pc);
                U64 a = t == KNIGHT ? KnightAtt[from] : t == BISHOP ? bishopAtt(from, occ) : t == ROOK ? rookAtt(from, occ) : queenAtt(from, occ);
                a &= target & ~ours;
                while (a) { int to = popLsb(a); *l++ = mk(from, to, p.board[to] != NO_PIECE ? CAPTURE : QUIET); }
            }
        }
        // pawns
        U64 pw = p.pieces(us, PAWN); int up = us == WHITE ? 8 : -8; U64 promoRank = us == WHITE ? RankBB[7] : RankBB[0], dbl = us == WHITE ? RankBB[2] : RankBB[5];
        U64 empty = ~occ;
        if (mode != G_NOISY || true) {
            U64 one = shiftUp(us, pw) & empty;
            U64 two = shiftUp(us, one & dbl) & empty;
            U64 o1 = one & target, o2 = two & target;
            U64 o1p = o1 & promoRank, o1n = o1 & ~promoRank;
            if (mode == G_QUIET || evas) { U64 b = o1n; while (b) { int to = popLsb(b); *l++ = mk(to - up, to, QUIET); } b = o2; while (b) { int to = popLsb(b); *l++ = mk(to - 2 * up, to, DPUSH); } }
            if (mode == G_NOISY) { // push-promotions: queen only. (target = enemy pieces, so use raw squares)
                U64 b = one & promoRank; while (b) { int to = popLsb(b); *l++ = mk(to - up, to, PROMO_Q); } }
            else { U64 b = o1p; while (b) { int to = popLsb(b); l = addPawnMoves(l, to - up, to, 0, false, mode); } }
        }
        if (mode != G_QUIET) {
            U64 tg = evas ? (target & theirs) : theirs;
            U64 cl = (us == WHITE ? ((pw & ~FILE_A) << 7) : ((pw & ~FILE_A) >> 9)) & tg, cr = (us == WHITE ? ((pw & ~(FILE_A << 7)) << 9) : ((pw & ~(FILE_A << 7)) >> 7)) & tg;
            int dl = us == WHITE ? 7 : -9, dr = us == WHITE ? 9 : -7;
            while (cl) { int to = popLsb(cl); if (bit(to) & promoRank) l = addPawnMoves(l, to - dl, to, 0, true, mode); else *l++ = mk(to - dl, to, CAPTURE); }
            while (cr) { int to = popLsb(cr); if (bit(to) & promoRank) l = addPawnMoves(l, to - dr, to, 0, true, mode); else *l++ = mk(to - dr, to, CAPTURE); }
            if (s.ep != NO_SQ) { U64 f = PawnAtt[them][s.ep] & pw; while (f) *l++ = mk(popLsb(f), s.ep, EP); }
        }
    }
    // castling
    if (mode == G_QUIET && !s.checkers) {
        int base = us == WHITE ? 0 : 56; int c = s.castling;
        if (c & (us == WHITE ? 1 : 4)) { if (!(occ & (bit(base + 5) | bit(base + 6))) && !p.attacked(base + 5, them, occ) && !p.attacked(base + 6, them, occ)) *l++ = mk(base + 4, base + 6, OO); }
        if (c & (us == WHITE ? 2 : 8)) { if (!(occ & (bit(base + 1) | bit(base + 2) | bit(base + 3))) && !p.attacked(base + 3, them, occ) && !p.attacked(base + 2, them, occ)) *l++ = mk(base + 4, base + 2, OOO); }
    }
    return l;
}
static int genLegal(Position& p, Move* out) {
    Move tmp[256]; Move* e;
    if (p.inCheck()) e = genMoves(p, tmp, G_EVASION);
    else { e = genMoves(p, tmp, G_NOISY); e = genMoves(p, e, G_QUIET); }
    int n = 0; for (Move* m = tmp; m < e; m++) if (p.isLegal(*m)) out[n++] = *m; return n;
}
static std::string moveStr(Move m) {
    if (m == MOVE_NONE) return "0000";
    std::string s; s += char('a' + fileOf(mFrom(m))); s += char('1' + rankOf(mFrom(m))); s += char('a' + fileOf(mTo(m))); s += char('1' + rankOf(mTo(m)));
    if (isPromo(m)) s += "nbrq"[promoType(m) - 1]; return s;
}
static Move parseMove(Position& p, const std::string& str) {
    Move l[256]; int n = genLegal(p, l); for (int i = 0; i < n; i++) if (moveStr(l[i]) == str) return l[i]; return MOVE_NONE;
}
static U64 perft(Position& p, int d) {
    Move l[256]; int n = genLegal(p, l); if (d == 1) return n; U64 c = 0;
    for (int i = 0; i < n; i++) { p.doMove(l[i]); c += perft(p, d - 1); p.undoMove(); } return c;
}
// ------------------------------- KPK bitbase --------------------------------
static uint8_t KPKres[2 * 24 * 64 * 64 + 8];
enum { K_INVALID = 0, K_UNKNOWN = 1, K_DRAW = 2, K_WIN = 4 };
static inline int kidx(int us, int bk, int wk, int psq) { return wk | (bk << 6) | (us << 12) | (fileOf(psq) << 13) | ((6 - rankOf(psq)) << 15); }
static void initKPK() {
    const int N = 2 * 24 * 64 * 64;
    auto ksqW = [](int i) { return i & 63; }; auto ksqB = [](int i) { return (i >> 6) & 63; };
    auto pawnSq = [](int i) { return (6 - ((i >> 15) & 7)) * 8 + ((i >> 13) & 3); };
    for (int i = 0; i < N; i++) {
        int wk = ksqW(i), bk = ksqB(i), us = (i >> 12) & 1, ps = pawnSq(i), r = (i >> 15) & 7; uint8_t res;
        if (r > 5 || Dist[wk][bk] <= 1 || wk == ps || bk == ps || (us == WHITE && (PawnAtt[WHITE][ps] & bit(bk)))) res = K_INVALID;
        else if (us == WHITE && rankOf(ps) == 6 && wk != ps + 8 && (Dist[bk][ps + 8] > 1 || (KingAtt[wk] & bit(ps + 8)))) res = K_WIN;
        else if (us == BLACK && (!(KingAtt[bk] & ~(KingAtt[wk] | PawnAtt[WHITE][ps])) || (KingAtt[bk] & ~KingAtt[wk] & bit(ps)))) res = K_DRAW;
        else res = K_UNKNOWN;
        KPKres[i] = res;
    }
    bool changed = true;
    while (changed) {
        changed = false;
        for (int i = 0; i < N; i++) {
            if (KPKres[i] != K_UNKNOWN) continue;
            int wk = ksqW(i), bk = ksqB(i), us = (i >> 12) & 1, ps = pawnSq(i); int r = K_INVALID;
            if (us == WHITE) {
                U64 b = KingAtt[wk]; while (b) r |= KPKres[kidx(BLACK, bk, popLsb(b), ps)];
                if (rankOf(ps) < 6) r |= KPKres[kidx(BLACK, bk, wk, ps + 8)];
                if (rankOf(ps) == 1 && ps + 8 != wk && ps + 8 != bk) r |= KPKres[kidx(BLACK, bk, wk, ps + 16)];
                uint8_t n = (r & K_WIN) ? K_WIN : (r & K_UNKNOWN) ? K_UNKNOWN : K_DRAW;
                if (n != K_UNKNOWN) { KPKres[i] = n; changed = true; }
            } else {
                U64 b = KingAtt[bk]; while (b) r |= KPKres[kidx(WHITE, popLsb(b), wk, ps)];
                uint8_t n = (r & K_DRAW) ? K_DRAW : (r & K_UNKNOWN) ? K_UNKNOWN : K_WIN;
                if (n != K_UNKNOWN) { KPKres[i] = n; changed = true; }
            }
        }
    }
}
static bool kpkWin(int strong, int sk, int wk, int ps, int strongToMove) {
    if (strong == BLACK) { sk ^= 56; wk ^= 56; ps ^= 56; }
    if (fileOf(ps) >= 4) { sk ^= 7; wk ^= 7; ps ^= 7; }
    return KPKres[kidx(strongToMove ? WHITE : BLACK, wk, sk, ps)] == K_WIN;
}

// -------------------------------- Evaluation --------------------------------
static const int MatMG[6] = {82, 337, 365, 477, 1025, 0}, MatEG[6] = {94, 281, 297, 512, 936, 0};
static int MobN[2][9], MobB[2][14], MobR[2][15], MobQ[2][28];
static int PK[12] = {14, 3, 6, 4, 8, 6, 2, 2, 3, 5, 8, 100};   // div, weakring, chk N/B/R/Q, attw N/B/R/Q, eg-div, danger-cap-scale
enum { G_MOBN, G_MOBB, G_MOBR, G_MOBQ, G_OUTPOST, G_BPAIR, G_BADB, G_ROOKF, G_ROOK7, G_TRAP, G_KDANGER, G_SHIELD, G_CENTERK, G_THPAWN, G_THMINOR, G_HANG, G_PUSH, G_PASSED, G_SPACE, G_PAWNS, G_PSQ, G_MOPUP, NG };
static int PM[NG], PE[NG], PB[8], PMAT[10], PTEMPO[2], PSCALE = 100;
static const char* GNAMES[NG] = {"MOBN","MOBB","MOBR","MOBQ","OUTPOST","BPAIR","BADB","ROOKF","ROOK7","TRAP","KDANGER","SHIELD","CENTERK","THPAWN","THMINOR","HANG","PUSH","PASSED","SPACE","PAWNS","PSQ","MOPUP"};
static void tunedParams(); static void tunedK();
static void defaultParams() { for (int i = 0; i < NG; i++) PM[i] = PE[i] = 100; for (int i = 0; i < 8; i++) PB[i] = 0; for (int t = 0; t < 5; t++) { PMAT[t] = MatMG[t]; PMAT[5 + t] = MatEG[t]; } PTEMPO[0] = 14; PTEMPO[1] = 6; PSCALE = 100; tunedParams(); }


static void tunedParams() {
    PM[0] = 52; PE[0] = 180; /* MOBN */
    PM[1] = 52; PE[1] = 124; /* MOBB */
    PM[2] = 124; PE[2] = 68; /* MOBR */
    PM[3] = 60; PE[3] = 172; /* MOBQ */
    PM[4] = 196; PE[4] = 164; /* OUTPOST */
    PM[5] = 52; PE[5] = 156; /* BPAIR */
    PM[6] = 84; PE[6] = 116; /* BADB */
    PM[7] = 156; PE[7] = 68; /* ROOKF */
    PM[8] = 156; PE[8] = 204; /* ROOK7 */
    PM[9] = 4; PE[9] = 100; /* TRAP */
    PM[10] = 132; PE[10] = 164; /* KDANGER */
    PM[11] = 68; PE[11] = 100; /* SHIELD */
    PM[12] = 68; PE[12] = 100; /* CENTERK */
    PM[13] = 124; PE[13] = 180; /* THPAWN */
    PM[14] = 212; PE[14] = 196; /* THMINOR */
    PM[15] = 172; PE[15] = 212; /* HANG */
    PM[16] = 84; PE[16] = 140; /* PUSH */
    PM[17] = 28; PE[17] = 84; /* PASSED */
    PM[18] = 212; PE[18] = 100; /* SPACE */
    PM[19] = 52; PE[19] = 36; /* PAWNS */
    PM[20] = 92; PE[20] = 76; /* PSQ */
    PM[21] = 100; PE[21] = 100; /* MOPUP */
    PB[0] = 12;
    PB[1] = 20;
    PB[2] = 12;
    PB[3] = 28;
    PB[4] = -16;
    PB[5] = 36;
    PB[6] = 12;
    PB[7] = 32;
    PMAT[0] = 82;
    PMAT[1] = 369;
    PMAT[2] = 389;
    PMAT[3] = 477;
    PMAT[4] = 1017;
    PMAT[5] = 78;
    PMAT[6] = 321;
    PMAT[7] = 305;
    PMAT[8] = 536;
    PMAT[9] = 1016;
    PTEMPO[0] = 47; PTEMPO[1] = 30; PSCALE = 118;
}

static void initEval() {
    defaultParams();
    static const int nmg[9] = {-62, -53, -12, -4, 3, 13, 22, 28, 33}, neg[9] = {-81, -56, -31, -16, 5, 11, 17, 20, 25};
    static const int bmg[14] = {-48, -20, 16, 26, 38, 51, 55, 63, 63, 68, 81, 81, 91, 98}, beg[14] = {-59, -23, -3, 13, 24, 42, 54, 57, 65, 73, 78, 86, 88, 97};
    static const int rmg[15] = {-60, -20, 2, 3, 3, 11, 22, 31, 40, 40, 41, 48, 57, 57, 62}, reg[15] = {-78, -17, 23, 39, 70, 99, 103, 121, 134, 139, 158, 164, 168, 169, 172};
    static const int qmg[28] = {-30, -12, -8, -9, 20, 23, 23, 35, 38, 53, 64, 65, 65, 66, 67, 67, 72, 72, 77, 79, 93, 108, 108, 108, 110, 114, 114, 116};
    static const int qeg[28] = {-48, -30, -7, 19, 40, 55, 59, 75, 78, 96, 96, 100, 121, 127, 131, 133, 136, 141, 147, 150, 151, 168, 168, 171, 182, 182, 192, 219};
    for (int i = 0; i < 9; i++) { MobN[0][i] = nmg[i] * 3 / 4; MobN[1][i] = neg[i] * 3 / 4; }
    for (int i = 0; i < 14; i++) { MobB[0][i] = bmg[i] * 3 / 4; MobB[1][i] = beg[i] * 3 / 4; }
    for (int i = 0; i < 15; i++) { MobR[0][i] = rmg[i] * 3 / 4; MobR[1][i] = reg[i] * 3 / 4; }
    for (int i = 0; i < 28; i++) { MobQ[0][i] = qmg[i] * 3 / 4; MobQ[1][i] = qeg[i] * 3 / 4; }
    // piece-square tables (generated shapes) + material
    for (int t = 0; t < 6; t++) for (int s = 0; s < 64; s++) {
        int f = fileOf(s), r = rankOf(s);   // white perspective
        double cf = std::abs(2 * f - 7) / 2.0, cr = std::abs(2 * r - 7) / 2.0, d = std::max(cf, cr), d2 = (cf + cr) / 2.0;
        int mg = 0, eg = 0;
        switch (t) {
            case PAWN: {
                static const int rm[8] = {0, -6, -4, 2, 12, 26, 44, 0}, re[8] = {0, -4, -4, 2, 8, 22, 40, 0};
                int fc = (f == 3 || f == 4) ? 1 : (f == 2 || f == 5) ? 0 : -1;
                mg = rm[r] + (r >= 2 && r <= 4 ? fc * (r == 3 ? 9 : 5) : 0) + ((f == 0 || f == 7) ? -4 : 0) + ((f == 1 || f == 6) && r == 1 ? 0 : 0);
                if (r == 1 && (f == 3 || f == 4)) mg -= 8;      // undeveloped centre pawns
                if (r == 1 && (f == 5 || f == 6 || f == 1 || f == 2)) mg += 4;
                eg = re[r] + ((f == 0 || f == 7) ? -3 : 0);
                if (r == 0 || r == 7) mg = eg = 0; break; }
            case KNIGHT: mg = (int)(26 - 17 * d) - ((r == 0) ? 8 : 0); eg = (int)(18 - 12 * d); break;
            case BISHOP: mg = (int)(10 - 6 * d) + ((f == r || f + r == 7) ? 5 : 0) - (r == 0 && (f == 2 || f == 5) ? 0 : (r == 0 ? 6 : 0)); eg = (int)(8 - 5 * d); break;
            case ROOK: mg = (f == 3 || f == 4 ? 5 : (f == 2 || f == 5 ? 2 : 0)) + (r == 6 ? 8 : 0) - (r == 0 && (f == 0 || f == 7) ? 4 : 0); eg = (r == 6 ? 6 : 0) + (r == 0 ? -3 : 0); break;
            case QUEEN: mg = (int)(8 - 4 * d) - (r == 0 ? 6 : 0); eg = (int)(14 - 9 * d); break;
            case KING: {
                // middlegame: stay on back rank / corner; endgame: centralise
                mg = -22 * r - (int)(8 * (3.5 - cf)) * 0 + (r == 0 ? (f <= 2 || f >= 6 ? 26 : (f == 3 || f == 4 ? -18 : 6)) : 0) - (r > 0 ? (int)(6 * (3.5 - cf)) : 0);
                eg = (int)(32 - 19 * d2 * 1.0) - (int)(4 * (d)); break; }
        }
        int vmg = mg, veg = eg;
        PSQ[t][s] = S(vmg, veg);                      // white piece on s
        PSQ[t + 6][s ^ 56] = S(-vmg, -veg);           // black piece mirrored, negative (score kept from white's view)
    }
}

struct PawnEntry { U64 key; int sc; U64 passed[2], att[2]; };
static thread_local PawnEntry pawnTT[16384];

static const int ConnR[8] = {0, 7, 8, 12, 29, 48, 86, 0};
static void evalPawns(const Position& p, PawnEntry& e) {
    int total = 0;
    for (int c = 0; c < 2; c++) {
        U64 P = p.pieces(c, PAWN), E = p.pieces(c ^ 1, PAWN); int sc = 0, up = c == WHITE ? 8 : -8;
        e.passed[c] = 0; e.att[c] = pawnAttsBB(c, P);
        U64 b = P;
        while (b) {
            int s = popLsb(b), f = fileOf(s), rr = relRank(c, s);
            U64 adj = AdjFiles[f];
            bool opposed = (ForwardMask[c][s] & E) != 0, blocked = (E & bit(s + up)) != 0;
            U64 support = P & PawnAtt[c ^ 1][s], phalanx = P & adj & RankBB[rankOf(s)], neigh = P & adj;
            bool passed = !(PassedMask[c][s] & E) && !(ForwardMask[c][s] & P);
            if (passed) e.passed[c] |= bit(s);
            bool backward = !(P & adj & ~PassedMask[c][s]) && (blocked || (PawnAtt[c][s + up] & E)) && rr < 6;
            if (support | phalanx) {
                int v = ConnR[rr] * (4 + 2 * (phalanx != 0) - 2 * opposed - blocked) / 2 + 17 * popcnt(support);
                sc += S(v, v * (rr - 2) / 4 < 0 ? 0 : v * (rr - 2) / 4);
            } else if (!neigh) sc -= S(6, 12) + (opposed ? 0 : S(10, 20));
            else if (backward) sc -= S(8, 16) + (opposed ? 0 : S(9, 18));
            if (!support && (ForwardMask[c][s] & P)) sc -= S(9, 28);
        }
        total += c == WHITE ? sc : -sc;
    }
    e.sc = total;
}

static const int PassMG[8] = {0, 4, 8, 10, 34, 90, 160, 0}, PassEG[8] = {0, 14, 20, 32, 62, 120, 190, 0};
#define AttW_(t) (PK[5 + (t)])

struct EvalInfo { U64 att[2][6], attAll[2], att2[2]; };

static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : v > hi ? hi : v; }

static int kingShield(const Position& p, int c, int ksq, const PawnEntry& pe) {
    int f = clampi(fileOf(ksq), 1, 6), kr = relRank(c, ksq), sc = 0; U64 P = p.pieces(c, PAWN), E = p.pieces(c ^ 1, PAWN);
    for (int ff = f - 1; ff <= f + 1; ff++) {
        U64 mine = P & FileBB[ff], theirs = E & FileBB[ff]; int md = 0, ed = 0;
        if (mine) { U64 m = mine; int best = 9; while (m) { int s = popLsb(m); int d = relRank(c, s) - kr; if (d >= 0 && d < best) best = d; } if (best < 9) md = best; }
        if (theirs) { U64 m = theirs; int best = 9; while (m) { int s = popLsb(m); int d = relRank(c, s) - kr; if (d >= 0 && d < best) best = d; } if (best < 9) ed = best; }
        if (md == 0) { sc -= 20; if (!ed) sc -= 12; } else sc += md == 1 ? 12 : md == 2 ? 6 : -2;
        if (ed) { int d = ed; sc -= (md == 0) ? (d <= 3 ? 14 : 5) : (d == 2 ? 10 : d == 3 ? 5 : d == 1 ? 3 : 0); }
    }
    return sc;
}

static int scaleFactor(const Position& p, int strong, int egv) {
    int weak = strong ^ 1;
    int sp = popcnt(p.pieces(strong, PAWN)), wp = popcnt(p.pieces(weak, PAWN));
    int sN = popcnt(p.pieces(strong, KNIGHT)), sB = popcnt(p.pieces(strong, BISHOP)), sR = popcnt(p.pieces(strong, ROOK)), sQ = popcnt(p.pieces(strong, QUEEN));
    int wN = popcnt(p.pieces(weak, KNIGHT)), wB = popcnt(p.pieces(weak, BISHOP)), wR = popcnt(p.pieces(weak, ROOK)), wQ = popcnt(p.pieces(weak, QUEEN));
    int snp = sN * 320 + sB * 330 + sR * 500 + sQ * 900, wnp = wN * 320 + wB * 330 + wR * 500 + wQ * 900;
    if (sp == 0) {
        if (wnp == 0 && wp == 0) {                         // bare king: need real mating material
            if (sR == 0 && sQ == 0 && sN + sB <= 1) return 0;
            if (sR == 0 && sQ == 0 && sB == 0 && sN == 2) return 0;
            return 64;
        }
        if (snp - wnp <= 330) return 0;
        if (sQ && !wQ && wR && snp - wnp > 330 && wN + wB == 0 && sR + sN + sB == 0) return 40;
        if (snp - wnp <= 500) return 6;
        return 40;
    }
    // KBP vs K with wrong-coloured rook pawn
    if (sp >= 1 && sN == 0 && sR == 0 && sQ == 0 && sB == 1 && wnp == 0 && wp == 0) {
        U64 pw = p.pieces(strong, PAWN);
        if (!(pw & ~FileBB[0]) || !(pw & ~FileBB[7])) {
            int pf = (pw & FileBB[0]) ? 0 : 7, promo = strong == WHITE ? pf + 56 : pf, bsq = lsb(p.pieces(strong, BISHOP));
            bool bishopLight = (bit(bsq) & LIGHT_SQ) != 0, promoLight = (bit(promo) & LIGHT_SQ) != 0;
            if (bishopLight != promoLight && Dist[p.ksq(weak)][promo] <= 1) return 0;
        }
    }
    // opposite-coloured bishops
    if (sN == 0 && sR == 0 && sQ == 0 && sB == 1 && wN == 0 && wR == 0 && wQ == 0 && wB == 1) {
        bool oc = ((bit(lsb(p.pieces(strong, BISHOP))) & LIGHT_SQ) != 0) != ((bit(lsb(p.pieces(weak, BISHOP))) & LIGHT_SQ) != 0);
        if (oc) { int d = sp - wp; return d <= 0 ? 8 : d == 1 ? 18 : d == 2 ? 32 : 44; }
    }
    // pure rook endings with a one pawn edge are drawish
    if (sN + sB + sQ + wN + wB + wQ == 0 && sR == 1 && wR == 1 && sp - wp <= 1) return sp - wp <= 0 ? 44 : 52;
    // lone minor vs pawn-heavy cases
    if (sp == 1 && sN + sB + sR + sQ == 0 && wN + wB + wR + wQ == 0 && wp == 0) return 64;
    return std::min(64, 44 + 6 * sp);
}


static bool EVDBG = false;
static int evaluate(const Position& p) {
    const StateInfo& st = p.st[p.sp];
    int us = p.side;
    int cnt[2][6];
    for (int c = 0; c < 2; c++) for (int t = 0; t < 5; t++) cnt[c][t] = popcnt(p.pieces(c, t));
    int phase = std::min(24, cnt[0][KNIGHT] + cnt[1][KNIGHT] + cnt[0][BISHOP] + cnt[1][BISHOP] + 2 * (cnt[0][ROOK] + cnt[1][ROOK]) + 4 * (cnt[0][QUEEN] + cnt[1][QUEEN]));
    for (int strong = 0; strong < 2; strong++) {
        int weak = strong ^ 1;
        if (p.byColor[weak] == p.pieces(weak, KING) && p.byColor[strong] == (p.pieces(strong, KING) | p.pieces(strong, PAWN)) && cnt[strong][PAWN] == 1) {
            int ps = lsb(p.pieces(strong, PAWN)); bool w = kpkWin(strong, p.ksq(strong), p.ksq(weak), ps, us == strong);
            if (!w) return 0;
            int v = 650 + 40 * relRank(strong, ps);
            return us == strong ? v : -v;
        }
    }
    PawnEntry& pe = pawnTT[st.pawnKey & 16383];
    if (pe.key != st.pawnKey) { evalPawns(p, pe); pe.key = st.pawnKey; }
    EvalInfo ei; memset(&ei, 0, sizeof ei);
    int units[2] = {0, 0}, attackers[2] = {0, 0};
    U64 ring[2];
    for (int c = 0; c < 2; c++) { int k = p.ksq(c); ring[c] = KingAtt[k] | bit(k); ring[c] |= (c == WHITE ? ring[c] << 8 : ring[c] >> 8); ei.att[c][PAWN] = pe.att[c]; ei.att[c][KING] = KingAtt[k]; }
    U64 occ = p.occ;
    int G[2][NG]; memset(G, 0, sizeof G);
    for (int c = 0; c < 2; c++) {
        int them = c ^ 1; U64 P = p.pieces(c, PAWN);
        U64 lowOrBlocked = P & (RankBB[c == WHITE ? 1 : 6] | RankBB[c == WHITE ? 2 : 5] | shiftUp(them, occ));
        U64 mobArea = ~(lowOrBlocked | p.pieces(c, KING) | pe.att[them]);
        U64 theirRing = ring[them];
        U64 b = p.pieces(c, KNIGHT);
        while (b) { int s = popLsb(b); U64 a = KnightAtt[s]; ei.att2[c] |= ei.attAll[c] & a; ei.att[c][KNIGHT] |= a; ei.attAll[c] |= a;
            int m = popcnt(a & mobArea); G[c][G_MOBN] += S(MobN[0][m] + PB[0], MobN[1][m] + PB[1]);
            if (a & theirRing) { attackers[c]++; units[c] += AttW_(KNIGHT) * popcnt(a & theirRing); }
            int rr = relRank(c, s);
            if (rr >= 3 && rr <= 5 && !((PassedMask[c][s] & ~FileBB[fileOf(s)]) & p.pieces(them, PAWN))) G[c][G_OUTPOST] += (P & PawnAtt[them][s]) ? S(30, 16) : S(14, 8);
        }
        b = p.pieces(c, BISHOP);
        while (b) { int s = popLsb(b); U64 a = bishopAtt(s, occ ^ p.pieces(c, QUEEN)); ei.att2[c] |= ei.attAll[c] & a; ei.att[c][BISHOP] |= a; ei.attAll[c] |= a;
            int m = popcnt(bishopAtt(s, occ) & mobArea); G[c][G_MOBB] += S(MobB[0][m] + PB[2], MobB[1][m] + PB[3]);
            if (a & theirRing) { attackers[c]++; units[c] += AttW_(BISHOP) * popcnt(a & theirRing); }
            int rr = relRank(c, s);
            if (rr >= 3 && rr <= 5 && !((PassedMask[c][s] & ~FileBB[fileOf(s)]) & p.pieces(them, PAWN)) && (P & PawnAtt[them][s])) G[c][G_OUTPOST] += S(14, 8);
            U64 same = (bit(s) & LIGHT_SQ) ? LIGHT_SQ : DARK_SQ; int bp = popcnt(P & same); G[c][G_BADB] -= S(2 * bp, 3 * bp);
            if (bp >= 4 && (shiftUp(them, occ) & P & same)) G[c][G_BADB] -= S(6, 8);
        }
        b = p.pieces(c, ROOK);
        while (b) { int s = popLsb(b); U64 a = rookAtt(s, occ ^ p.pieces(c, ROOK) ^ p.pieces(c, QUEEN)); ei.att2[c] |= ei.attAll[c] & a; ei.att[c][ROOK] |= a; ei.attAll[c] |= a;
            int m = popcnt(rookAtt(s, occ) & mobArea); G[c][G_MOBR] += S(MobR[0][m] + PB[4], MobR[1][m] + PB[5]);
            if (a & theirRing) { attackers[c]++; units[c] += AttW_(ROOK) * popcnt(a & theirRing); }
            int f = fileOf(s); U64 fb = FileBB[f];
            if (!(P & fb)) G[c][G_ROOKF] += (p.pieces(them, PAWN) & fb) ? S(16, 6) : S(34, 18);
            int rr = relRank(c, s);
            if (rr == 6 && (relRank(c, p.ksq(them)) == 7 || (p.pieces(them, PAWN) & RankBB[c == WHITE ? 6 : 1]))) G[c][G_ROOK7] += S(16, 28);
            if (m <= 3) { int kf = fileOf(p.ksq(c)); if ((kf < 4) == (f < kf) && relRank(c, p.ksq(c)) == rr) G[c][G_TRAP] -= S(36, 4) * (1 + !(st.castling & (c == WHITE ? 3 : 12))); }
        }
        b = p.pieces(c, QUEEN);
        while (b) { int s = popLsb(b); U64 a = queenAtt(s, occ); ei.att2[c] |= ei.attAll[c] & a; ei.att[c][QUEEN] |= a; ei.attAll[c] |= a;
            int m = popcnt(a & mobArea); G[c][G_MOBQ] += S(MobQ[0][m] + PB[6], MobQ[1][m] + PB[7]);
            if (a & theirRing) { attackers[c]++; units[c] += AttW_(QUEEN) * popcnt(a & theirRing); }
        }
        ei.attAll[c] |= ei.att[c][PAWN] | ei.att[c][KING];
        if (cnt[c][BISHOP] >= 2) G[c][G_BPAIR] += S(30, 50);
    }
    for (int c = 0; c < 2; c++) {
        int them = c ^ 1, k = p.ksq(c); U64 P = p.pieces(c, PAWN), ours = p.byColor[c], theirs = p.byColor[them];
        int u = units[them], at = attackers[them];
        U64 weakSq = ei.attAll[them] & ~ei.att2[c] & (~ei.att[c][PAWN]) & ring[c] & ~ours;
        u += PK[1] * popcnt(weakSq & KingAtt[k]);
        U64 safe = ~theirs & (~ei.attAll[c] | (ei.att2[them] & ~ei.att2[c]));
        U64 rk = rookAtt(k, occ), bk = bishopAtt(k, occ), nk = KnightAtt[k];
        int checks = 0;
        if (nk & ei.att[them][KNIGHT] & safe) { u += PK[2]; checks++; }
        if (bk & ei.att[them][BISHOP] & safe) { u += PK[3]; checks++; }
        if (rk & ei.att[them][ROOK] & safe) { u += PK[4]; checks++; }
        if ((rk | bk) & ei.att[them][QUEEN] & safe) { u += PK[5]; checks++; }
        if (!cnt[them][QUEEN]) u = u * 2 / 3;
        int pen = 0;
        if (at >= 2 || (at >= 1 && cnt[them][QUEEN] && checks)) pen = std::min(650, u * u / PK[0]);
        G[c][G_KDANGER] -= S(pen, pen / PK[10]);
        G[c][G_SHIELD] += S(kingShield(p, c, k, pe), 0);
        if (relRank(c, k) == 0 && (fileOf(k) == 3 || fileOf(k) == 4) && cnt[them][QUEEN]) G[c][G_CENTERK] -= (st.castling & (c == WHITE ? 3 : 12)) ? S(10, 0) : S(34, 0);
        U64 theirNP = theirs & ~p.pieces(them, PAWN) & ~p.pieces(them, KING);
        U64 minorAtt = ei.att[c][KNIGHT] | ei.att[c][BISHOP];
        G[c][G_THPAWN] += popcnt(ei.att[c][PAWN] & theirNP) * S(50, 28);
        G[c][G_THMINOR] += popcnt(minorAtt & (p.pieces(them, ROOK) | p.pieces(them, QUEEN)) & ~ei.att[c][PAWN]) * S(34, 20) + popcnt(ei.att[c][ROOK] & p.pieces(them, QUEEN)) * S(30, 20);
        U64 hang = theirNP & ~ei.attAll[them] & ei.attAll[c]; U64 undefPawns = p.pieces(them, PAWN) & ~ei.attAll[them] & ei.attAll[c];
        G[c][G_HANG] += popcnt(hang) * S(18, 12) + popcnt(undefPawns) * S(6, 8);
        U64 push = shiftUp(c, P) & ~occ & ~ei.att[them][PAWN] & (ei.attAll[c] | ~ei.attAll[them]);
        G[c][G_PUSH] += popcnt(pawnAttsBB(c, push) & theirNP) * S(16, 10);
        U64 pp = pe.passed[c]; int up = c == WHITE ? 8 : -8;
        while (pp) {
            int s = popLsb(pp), rr = relRank(c, s), stop = s + up;
            int mg = PassMG[rr], eg = PassEG[rr];
            if (rr >= 3) {
                int w = 5 * rr - 13;
                eg += (Dist[stop][p.ksq(them)] * 19 / 4 - Dist[stop][k] * 2) * w / 4;
                U64 path = ForwardMask[c][s]; bool clear = !(path & occ);
                if (clear) { bool safeAll = !(path & ei.attAll[them]); bool defAll = !(path & ~ei.attAll[c]);
                    int k2 = safeAll ? 35 : (!(path & ei.attAll[them] & ~ei.attAll[c]) || defAll ? 20 : 8); mg += k2 * rr / 6; eg += k2 * rr / 3; }
                else if (p.board[stop] != NO_PIECE && colorOf(p.board[stop]) != c) { mg -= 4 * rr; eg -= 6 * rr; }
                if (p.byColor[them] == (p.pieces(them, KING) | p.pieces(them, PAWN)) && clear) {
                    int promo = c == WHITE ? (fileOf(s) + 56) : fileOf(s), mv = 7 - rr - (rr == 1 ? 1 : 0);
                    int dk = Dist[p.ksq(them)][promo]; bool theirMove = (p.side == them);
                    if (dk > mv + (theirMove ? 1 : 0)) eg += 380 + 30 * rr;
                }
            }
            U64 behind = ForwardMask[them][s] & p.pieces(c, ROOK); if (behind && !(BetweenBB[s][lsb(behind)] & occ)) { eg += 18; mg += 6; }
            U64 ebehind = ForwardMask[them][s] & p.pieces(them, ROOK); if (ebehind && !(BetweenBB[s][lsb(ebehind)] & occ)) eg -= 10;
            G[c][G_PASSED] += S(mg, eg);
        }
        U64 spaceMask = (c == WHITE ? (RankBB[1] | RankBB[2] | RankBB[3]) : (RankBB[6] | RankBB[5] | RankBB[4])) & (FileBB[2] | FileBB[3] | FileBB[4] | FileBB[5]);
        U64 safeSp = spaceMask & ~P & ~ei.att[them][PAWN] & (~ei.attAll[them] | ei.attAll[c]);
        int w = std::max(0, popcnt(ours & ~P) - 3);
        G[c][G_SPACE] += S((popcnt(safeSp) + popcnt(safeSp & shiftUp(them, P))) * w * w / 16, 0);
    }
    for (int strong = 0; strong < 2; strong++) {
        int weak = strong ^ 1;
        if (p.byColor[weak] == p.pieces(weak, KING)) {
            int wk = p.ksq(weak), sk = p.ksq(strong);
            int cd = (std::abs(2 * fileOf(wk) - 7) + std::abs(2 * rankOf(wk) - 7)) / 2;
            int bonus = 28 * cd + 18 * (7 - Dist[wk][sk]);
            if (cnt[strong][BISHOP] >= 1 && cnt[strong][KNIGHT] >= 1 && !cnt[strong][PAWN] && !cnt[strong][ROOK] && !cnt[strong][QUEEN]) {
                bool light = (p.pieces(strong, BISHOP) & LIGHT_SQ) != 0; int cd2 = light ? std::min(Dist[wk][A8], Dist[wk][H1]) : std::min(Dist[wk][A1], Dist[wk][H8]);
                bonus = 40 * (7 - cd2) + 18 * (7 - Dist[wk][sk]);
            }
            G[strong][G_MOPUP] += S(0, bonus);
        }
    }
    // material (tunable) + positional psq (st.psq holds only positional part)
    int matMg = 0, matEg = 0;
    for (int t = 0; t < 5; t++) { int d = cnt[0][t] - cnt[1][t]; matMg += d * PMAT[t]; matEg += d * PMAT[5 + t]; }
    int mg = matMg, eg = matEg;
    G[0][G_PSQ] = st.psq; G[1][G_PSQ] = 0;
    G[0][G_PAWNS] = pe.sc; G[1][G_PAWNS] = 0;
    for (int g = 0; g < NG; g++) {
        int d = G[0][g] - G[1][g];
        if (g == G_PSQ || g == G_PAWNS) d = G[0][g];
        int gm = mgOf(G[0][g]) - mgOf(G[1][g]), ge = egOf(G[0][g]) - egOf(G[1][g]);
        if (g == G_PSQ || g == G_PAWNS) { gm = mgOf(G[0][g]); ge = egOf(G[0][g]); }
        (void)d;
        if (EVDBG) printf("  %-8s mg %5d eg %5d\n", GNAMES[g], gm * PM[g] / 100, ge * PE[g] / 100);
        mg += gm * PM[g] / 100; eg += ge * PE[g] / 100;
    }
    int strong = eg > 0 ? WHITE : BLACK, scale = scaleFactor(p, strong, eg);
    int v = (mg * phase + eg * (24 - phase) * scale / 64) / 24;
    v = v * PSCALE / 100;
    v += us == WHITE ? (PTEMPO[0] * phase + PTEMPO[1] * (24 - phase)) / 24 : -(PTEMPO[0] * phase + PTEMPO[1] * (24 - phase)) / 24;
    return us == WHITE ? v : -v;
}
// --------------------------- Transposition table ----------------------------
enum Bound { B_NONE = 0, B_UPPER = 1, B_LOWER = 2, B_EXACT = 3 };
static const int VALUE_MATE = 32000, VALUE_INF = 32500, MAX_PLY = 128, VALUE_NONE = 32767;
static const int MATE_IN_MAX = VALUE_MATE - MAX_PLY, VALUE_WIN = 10000;
struct TTEntry { U64 kx, data; };   // kx = key ^ data (lock-free consistency check)
struct TTData { Move move; int value, eval, depth, bound; bool pv; int gen; };
struct TT {
    TTEntry* tab = nullptr; size_t nb = 0; uint8_t gen = 0;
    ~TT() { free(tab); }
    void resize(size_t mb) {
        free(tab); nb = std::max<size_t>(1, mb * 1024 * 1024 / 64);
        tab = (TTEntry*)calloc(nb * 4, sizeof(TTEntry)); if (!tab) { nb = 1024; tab = (TTEntry*)calloc(nb * 4, sizeof(TTEntry)); }
    }
    void clear() { memset(tab, 0, nb * 4 * sizeof(TTEntry)); gen = 0; }
    void newSearch() { gen = (gen + 8) & 0xF8; }
    TTEntry* bucket(U64 key) const { return &tab[(size_t)(((unsigned __int128)key * nb) >> 64) * 4]; }
    static U64 pack(Move m, int v, int e, int d, int b, bool pv, int g) {
        return (U64)m | ((U64)(uint16_t)(int16_t)v << 16) | ((U64)(uint16_t)(int16_t)e << 32) | ((U64)(uint8_t)(d + 8) << 48) | ((U64)(g | (pv ? 4 : 0) | b) << 56);
    }
    bool probe(U64 key, TTData& d) const {
        TTEntry* b = bucket(key);
        for (int i = 0; i < 4; i++) {
            U64 dat = b[i].data, kx = b[i].kx;
            if (dat && (kx ^ dat) == key) {
                d.move = (Move)(dat & 0xFFFF); d.value = (int16_t)((dat >> 16) & 0xFFFF); d.eval = (int16_t)((dat >> 32) & 0xFFFF);
                d.depth = (int)((dat >> 48) & 0xFF) - 8; int fl = (int)(dat >> 56); d.bound = fl & 3; d.pv = fl & 4; d.gen = fl & 0xF8; return true;
            }
        }
        return false;
    }
    void store(U64 key, Move m, int v, int e, int depth, int bound, bool pv) {
        TTEntry* b = bucket(key); TTEntry* rep = nullptr; int worst = 1 << 30;
        for (int i = 0; i < 4; i++) {
            U64 dat = b[i].data;
            if (!dat) { rep = &b[i]; worst = -1; break; }
            if ((b[i].kx ^ dat) == key) {
                // keep old move if no new one
                Move om = (Move)(dat & 0xFFFF); int od = (int)((dat >> 48) & 0xFF) - 8; int ob = (int)(dat >> 56) & 3;
                if (!m) m = om;
                if (bound != B_EXACT && od > depth + 3 && (int)((dat >> 56) & 0xF8) == gen && ob != B_NONE) { if (od - depth > 4) return; }
                rep = &b[i]; worst = -2; break;
            }
            int od = (int)((dat >> 48) & 0xFF) - 8, ag = (gen - (int)((dat >> 56) & 0xF8)) & 0xF8;
            int w = od - 2 * ag; if (w < worst) { worst = w; rep = &b[i]; }
        }
        U64 d = pack(m, v, e, depth, bound, pv, gen); rep->data = d; rep->kx = key ^ d;
    }
    int hashfull() const {
        int c = 0; size_t n = std::min<size_t>(nb, 250);
        for (size_t i = 0; i < n; i++) for (int j = 0; j < 4; j++) { U64 d = tab[i * 4 + j].data; if (d && (((int)(d >> 56) & 0xF8) == gen)) c++; }
        return (int)(c * 1000 / (n * 4));
    }
};
static TT TTable;
static inline int valueToTT(int v, int ply) { return v >= MATE_IN_MAX ? v + ply : v <= -MATE_IN_MAX ? v - ply : v; }
static inline int valueFromTT(int v, int ply) { return v == VALUE_NONE ? v : v >= MATE_IN_MAX ? v - ply : v <= -MATE_IN_MAX ? v + ply : v; }

// ------------------------------- Limits / time ------------------------------
struct Limits { int64_t time[2] = {0, 0}, inc[2] = {0, 0}; int movestogo = 0; int64_t movetime = 0; int depth = 0; U64 nodes = 0; bool infinite = false; int mate = 0; std::vector<Move> searchmoves; };
static std::atomic<bool> gStop(false);
static int gOverhead = 30, gMultiPV = 1, gThreads = 1; static bool gQuiet = false;
static Limits gLim; static int64_t gOptimum = 0, gMaximum = 0;
static std::chrono::steady_clock::time_point gStart;
static inline int64_t elapsedMs() { return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - gStart).count(); }

// --------------------------------- Search -----------------------------------
static int Reductions[64][64];
static void initSearch() {
    for (int d = 1; d < 64; d++) for (int m = 1; m < 64; m++) Reductions[d][m] = (int)(0.70 + std::log((double)d) * std::log((double)m) / 2.15);
}
static const int CONT_N = 13 * 64;
struct SS { Move move; int staticEval; int ply; Move killers[2]; Move excluded; int16_t* cont; bool inCheck; bool ttPv; int doubleExt; int currentPieceTo; };
struct RootMove { Move m; int score = -VALUE_INF, prev = -VALUE_INF; int seldepth = 0; std::vector<Move> pv; U64 nodes = 0; };

static inline void gravity(int16_t& h, int bonus, int lim = 16384) { h += bonus - (int)h * std::abs(bonus) / lim; }

struct Thread {
    int id = 0; Position pos; U64 nodes = 0; int seldepth = 0, rootDepth = 0, completed = 0, pvIdx = 0, nmpMinPly = 0, rootDelta = 1;
    std::vector<RootMove> root; SS stack[MAX_PLY + 16]; Move pvTable[MAX_PLY + 2][MAX_PLY + 2]; int pvLen[MAX_PLY + 2];
    std::vector<int16_t> mainH, capH, contH, corrP, corrN; std::vector<Move> counter;
    int bestMoveChanges = 0; U64 tbNodes = 0;
    void alloc() {
        mainH.assign(2 * 4096, 0); capH.assign(12 * 64 * 6, 0); contH.assign((size_t)CONT_N * 768, 0); corrP.assign(2 * 16384, 0); corrN.assign(2 * 2 * 16384, 0); counter.assign(768, 0);
    }
    void clearHist() { std::fill(mainH.begin(), mainH.end(), 0); std::fill(capH.begin(), capH.end(), 0); std::fill(contH.begin(), contH.end(), 0); std::fill(corrP.begin(), corrP.end(), 0); std::fill(corrN.begin(), corrN.end(), 0); std::fill(counter.begin(), counter.end(), 0); }

    int16_t& mainHist(int c, Move m) { return mainH[c * 4096 + mFrom(m) * 64 + mTo(m)]; }
    int16_t& capHist(int pc, int to, int vt) { return capH[(pc * 64 + to) * 6 + vt]; }

    bool timeUp() {
        if (gStop.load(std::memory_order_relaxed)) return true;
        if (id != 0) return false;
        if (gLim.nodes && nodes >= gLim.nodes) { gStop = true; return true; }
        if (rootDepth > 1 && (gLim.movetime || gLim.time[0] || gLim.time[1]) && !gLim.infinite) { if (elapsedMs() >= gMaximum) { gStop = true; return true; } }
        return false;
    }
    int correction(const Position& p) {
        const StateInfo& s = p.st[p.sp]; int c = p.side;
        int v = corrP[c * 16384 + (s.pawnKey & 16383)] * 2 + corrN[(c * 2 + 0) * 16384 + (s.npKey[0] & 16383)] + corrN[(c * 2 + 1) * 16384 + (s.npKey[1] & 16383)];
        return v / 256;
    }
    void updateCorrection(const Position& p, int diff, int depth) {
        const StateInfo& s = p.st[p.sp]; int c = p.side;
        int w = std::min(16, depth + 1), tgt = clampi(diff, -100, 100) * 64;
        auto upd = [&](int16_t& e) { e = (int16_t)(e + (tgt - e) * w / 256); };
        upd(corrP[c * 16384 + (s.pawnKey & 16383)]); upd(corrN[(c * 2 + 0) * 16384 + (s.npKey[0] & 16383)]); upd(corrN[(c * 2 + 1) * 16384 + (s.npKey[1] & 16383)]);
    }

    int qsearch(int alpha, int beta, int ply, int qd);
    int search(int alpha, int beta, int depth, int ply, bool cutNode);
    void idLoop();
    bool hasNonPawn(int c) const { return (pos.byColor[c] & ~pos.byType[PAWN] & ~pos.byType[KING]) != 0; }
};

// move picker -----------------------------------------------------------------
struct MovePicker {
    Thread& th; Position& pos; Move tt, k1, k2, cm; int stage = 0; Move mv[256]; int sc[256]; int n = 0, idx = 0; Move bad[128]; int nbad = 0, bidx = 0; bool skipQuiets = false, evas, qs;
    int16_t *c1, *c2, *c4; int ply;
    MovePicker(Thread& t, Move ttm, Move a, Move b, Move c, bool evasion, bool q, SS* ss) : th(t), pos(t.pos), tt(ttm), k1(a), k2(b), cm(c), evas(evasion), qs(q) {
        c1 = (ss - 1)->cont; c2 = (ss - 2)->cont; c4 = (ss - 4)->cont; ply = ss->ply; stage = 0; if (!tt) stage = 1; }
    int noisyScore(Move m) {
        int to = mTo(m), fl = mFlag(m); int vt = fl == EP ? PAWN : (pos.board[to] == NO_PIECE ? NO_TYPE : typeOf(pos.board[to]));
        int s = 0; if (vt != NO_TYPE) s = PieceVal[vt] * 10 - PieceVal[typeOf(pos.board[mFrom(m)])] / 10 * 0 + th.capHist(pos.board[mFrom(m)], to, vt) / 16;
        if (fl & 8) s += (promoType(m) == QUEEN ? 8000 : -3000);
        return s;
    }
    int quietScore(Move m) {
        int pc = pos.board[mFrom(m)], to = mTo(m); int idx = pc * 64 + to;
        int s = th.mainHist(pos.side, m) * 2 + c1[idx] * 2 + c2[idx] + c4[idx] / 2;
        return s;
    }
    Move pickBest() {
        int bi = idx; for (int i = idx + 1; i < n; i++) if (sc[i] > sc[bi]) bi = i;
        std::swap(mv[idx], mv[bi]); std::swap(sc[idx], sc[bi]); return mv[idx++];
    }
    Move next() {
        for (;;) {
            switch (stage) {
            case 0: stage = 1; return tt;
            case 1: {
                if (evas) { Move* e = genMoves(pos, mv, G_EVASION); n = (int)(e - mv); idx = 0; int j = 0;
                    for (int i = 0; i < n; i++) { Move m = mv[i]; if (m == tt) continue; if (!pos.isLegal(m)) continue; mv[j] = m;
                        sc[j] = isNoisy(m) ? 100000 + noisyScore(m) : quietScore(m); j++; } n = j; stage = 20; continue; }
                Move* e = genMoves(pos, mv, G_NOISY); n = (int)(e - mv); idx = 0; int j = 0;
                for (int i = 0; i < n; i++) { Move m = mv[i]; if (m == tt) continue; if (!pos.isLegal(m)) continue; mv[j] = m; sc[j] = noisyScore(m); j++; } n = j;
                stage = qs ? 21 : 2; continue; }
            case 2: { // good noisy
                while (idx < n) { int s = sc[idx]; Move m = pickBest(); (void)s;
                    if (pos.seeGE(m, -(sc[idx - 1] > 0 ? 0 : 0))) return m; bad[nbad++] = m; }
                stage = 3; continue; }
            case 3: stage = 4; if (k1 && k1 != tt && !isNoisy(k1) && pos.isPseudoLegal(k1)) return k1; continue;
            case 4: stage = 5; if (k2 && k2 != tt && k2 != k1 && !isNoisy(k2) && pos.isPseudoLegal(k2)) return k2; continue;
            case 5: stage = 6; if (cm && cm != tt && cm != k1 && cm != k2 && !isNoisy(cm) && pos.isPseudoLegal(cm)) return cm; continue;
            case 6: {
                if (skipQuiets) { stage = 8; continue; }
                Move* e = genMoves(pos, mv, G_QUIET); n = (int)(e - mv); idx = 0; int j = 0;
                for (int i = 0; i < n; i++) { Move m = mv[i]; if (m == tt || m == k1 || m == k2 || m == cm) continue; if (!pos.isLegal(m)) continue; mv[j] = m; sc[j] = quietScore(m); j++; } n = j; stage = 7; continue; }
            case 7: if (skipQuiets) { stage = 8; continue; } if (idx < n) return pickBest(); stage = 8; continue;
            case 8: if (bidx < nbad) return bad[bidx++]; return MOVE_NONE;
            case 20: if (idx < n) return pickBest(); return MOVE_NONE;
            case 21: if (idx < n) return pickBest(); return MOVE_NONE;
            }
        }
    }
};

static inline int drawScore(U64 nodes) { return 0 - 1 + (int)(nodes & 2); }

int Thread::qsearch(int alpha, int beta, int ply, int qd) {
    nodes++; if (ply > seldepth) seldepth = ply;
    if ((nodes & 1023) == 0 && timeUp()) return 0;
    if (gStop.load(std::memory_order_relaxed)) return 0;
    bool inCheck = pos.inCheck();
    if (pos.isRepetition(ply) || pos.st[pos.sp].halfmove >= 100 || pos.insufficient()) {
        if (pos.st[pos.sp].halfmove >= 100 && inCheck) { Move l[256]; if (!genLegal(pos, l)) return -VALUE_MATE + ply; }
        return drawScore(nodes);
    }
    if (ply >= MAX_PLY - 1) return inCheck ? 0 : evaluate(pos);
    TTData t{}; U64 key = pos.key(); bool found = TTable.probe(key, t);
    int ttValue = found ? valueFromTT(t.value, ply) : VALUE_NONE; Move ttMove = found ? t.move : MOVE_NONE;
    bool pvNode = beta - alpha > 1;
    if (!pvNode && found && ttValue != VALUE_NONE && t.depth >= (inCheck ? 0 : -1) &&
        ((t.bound & B_LOWER && ttValue >= beta) || (t.bound & B_UPPER && ttValue <= alpha) || t.bound == B_EXACT)) return ttValue;
    int best, staticEval = VALUE_NONE;
    if (inCheck) best = -VALUE_INF;
    else {
        int raw = found && t.eval != VALUE_NONE ? t.eval : evaluate(pos);
        staticEval = raw + correction(pos); best = staticEval;
        if (found && ttValue != VALUE_NONE && ((t.bound & B_LOWER && ttValue > best) || (t.bound & B_UPPER && ttValue < best) || t.bound == B_EXACT)) best = ttValue;
        if (best >= beta) { if (!found) TTable.store(key, MOVE_NONE, valueToTT(best, ply), raw, -2, B_LOWER, false); return best; }
        if (best > alpha) alpha = best;
        if (!found) TTable.store(key, MOVE_NONE, VALUE_NONE, raw, -2, B_NONE, false);
    }
    SS* ss = &stack[ply + 8]; (ss)->ply = ply; ss->inCheck = inCheck;
    MovePicker mp(*this, (ttMove && pos.isPseudoLegal(ttMove) && (inCheck || isNoisy(ttMove))) ? ttMove : MOVE_NONE, 0, 0, 0, inCheck, true, ss);
    int moveCount = 0, bound = B_UPPER; Move bestMove = MOVE_NONE; int futBase = staticEval + 140;
    Move m;
    while ((m = mp.next()) != MOVE_NONE) {
        moveCount++;
        bool cap = isCap(m);
        if (!inCheck) {
            if (!(mFlag(m) & 8) && best > -MATE_IN_MAX) {
                int vv = mFlag(m) == EP ? PieceVal[PAWN] : PieceVal[typeOf(pos.board[mTo(m)])];
                if (futBase + vv <= alpha && !pos.seeGE(m, 1)) { best = std::max(best, futBase + vv - 0); continue; }
                if (!pos.seeGE(m, 0)) continue;
            }
            (void)cap;
        }
        int pcTo = pos.board[mFrom(m)] * 64 + mTo(m);
        ss->move = m; ss->cont = &contH[(size_t)pcTo * 768];
        pos.doMove(m);
        int v = -qsearch(-beta, -alpha, ply + 1, qd + 1);
        pos.undoMove();
        if (gStop.load(std::memory_order_relaxed)) return 0;
        if (v > best) { best = v; if (v > alpha) { bestMove = m; if (v >= beta) { bound = B_LOWER; break; } alpha = v; bound = B_EXACT; } }
    }
    if (inCheck && best == -VALUE_INF) return -VALUE_MATE + ply;
    TTable.store(key, bestMove, valueToTT(best, ply), staticEval == VALUE_NONE ? VALUE_NONE : staticEval - correction(pos), inCheck ? 0 : -1, bound, pvNode);
    return best;
}
// ------------------------------ Main search ---------------------------------
int Thread::search(int alpha, int beta, int depth, int ply, bool cutNode) {
    bool pv = beta - alpha > 1, rootNode = ply == 0;
    if (depth <= 0) return qsearch(alpha, beta, ply, 0);
    nodes++;
    if (pv) { pvLen[ply] = ply; if (ply > seldepth) seldepth = ply; }
    if ((nodes & 1023) == 0 && timeUp()) return 0;
    if (gStop.load(std::memory_order_relaxed)) return 0;
    SS* ss = &stack[ply + 8]; ss->ply = ply; (ss + 1)->excluded = MOVE_NONE; ss->doubleExt = rootNode ? 0 : (ss - 1)->doubleExt;
    bool inCheck = pos.inCheck(); Move excluded = ss->excluded;
    if (!rootNode) {
        if (pos.isRepetition(ply) || pos.st[pos.sp].halfmove >= 100 || pos.insufficient()) {
            if (pos.st[pos.sp].halfmove >= 100 && inCheck) { Move l[256]; if (!genLegal(pos, l)) return -VALUE_MATE + ply; }
            return drawScore(nodes);
        }
        if (ply >= MAX_PLY - 1) return inCheck ? 0 : evaluate(pos);
        alpha = std::max(alpha, -VALUE_MATE + ply); beta = std::min(beta, VALUE_MATE - ply - 1);
        if (alpha >= beta) return alpha;
    }
    ss->inCheck = inCheck;
    (ss + 2)->killers[0] = (ss + 2)->killers[1] = 0;
    U64 key = pos.key() ^ (excluded ? ((U64)excluded * 0x9E3779B97F4A7C15ULL) : 0);
    TTData t{}; bool found = TTable.probe(key, t);
    int ttValue = found ? valueFromTT(t.value, ply) : VALUE_NONE;
    Move ttMove = rootNode ? (pvIdx < (int)root.size() ? root[pvIdx].m : MOVE_NONE) : (found ? t.move : MOVE_NONE);
    if (ttMove && !rootNode && !pos.isPseudoLegal(ttMove)) ttMove = MOVE_NONE;
    bool ttCapture = ttMove && isNoisy(ttMove);
    ss->ttPv = pv || (found && t.pv);
    if (!pv && !excluded && found && ttValue != VALUE_NONE && t.depth >= depth && pos.st[pos.sp].halfmove < 90 &&
        ((t.bound & B_LOWER && ttValue >= beta) || (t.bound & B_UPPER && ttValue <= alpha) || t.bound == B_EXACT)) {
        if (ttMove && ttValue >= beta && !isNoisy(ttMove) && !inCheck) {
            int bonus = std::min(150 * depth - 80, 1600); gravity(mainHist(pos.side, ttMove), bonus);
        }
        return ttValue;
    }
    int staticEval = VALUE_NONE, eval = VALUE_NONE; bool improving = false;
    if (inCheck) { ss->staticEval = VALUE_NONE; }
    else {
        if (excluded) { staticEval = eval = ss->staticEval; }
        else {
            int raw = found && t.eval != VALUE_NONE ? t.eval : evaluate(pos);
            staticEval = raw + correction(pos); ss->staticEval = eval = staticEval;
            if (found && ttValue != VALUE_NONE && ((t.bound & B_LOWER && ttValue > eval) || (t.bound & B_UPPER && ttValue < eval) || t.bound == B_EXACT)) eval = ttValue;
            if (!found) TTable.store(key, MOVE_NONE, VALUE_NONE, raw, -2, B_NONE, ss->ttPv);
        }
        int pe = (ss - 2)->staticEval; if ((ss - 2)->inCheck || pe == VALUE_NONE) { pe = (ss - 4)->staticEval; if ((ss - 4)->inCheck) pe = VALUE_NONE; }
        improving = pe != VALUE_NONE && ss->staticEval > pe;
    }
    int us = pos.side;
    // ---- node-level pruning ----
    if (!rootNode && !inCheck && !excluded) {
        if (!pv && eval < alpha - 330 - 230 * depth * depth && depth <= 3) { int v = qsearch(alpha - 1, alpha, ply, 0); if (v < alpha) return v; }
        if (!pv && depth <= 9 && eval >= beta && eval - (88 * depth - 55 * improving) >= beta && eval < MATE_IN_MAX && beta > -MATE_IN_MAX && !(ttMove && ttCapture && false)) return (eval + beta) / 2;
        if (!pv && (ss - 1)->move != MOVE_NONE && eval >= beta && staticEval >= beta - 18 * depth + 190 && hasNonPawn(us) && beta > -MATE_IN_MAX && ply >= nmpMinPly) {
            int R = 4 + depth / 3 + std::min((eval - beta) / 190, 3);
            ss->move = MOVE_NONE; ss->cont = &contH[(size_t)(12 * 64) * 768];
            pos.doNull(); int v = -search(-beta, -beta + 1, depth - R, ply + 1, !cutNode); pos.undoNull();
            if (gStop.load(std::memory_order_relaxed)) return 0;
            if (v >= beta && v < MATE_IN_MAX) {
                if (nmpMinPly || depth < 14) return v;
                nmpMinPly = ply + 3 * (depth - R) / 4;
                int v2 = search(beta - 1, beta, depth - R, ply, false); nmpMinPly = 0;
                if (v2 >= beta) return v;
            }
        }
    }
    // internal iterative reduction
    if (!inCheck && depth >= 4 && !ttMove && (pv || cutNode)) depth--;
    // ---- probcut ----
    int probBeta = beta + 190 - 55 * improving;
    if (!pv && !inCheck && !excluded && depth >= 5 && std::abs(beta) < MATE_IN_MAX && !(found && t.depth >= depth - 3 && ttValue != VALUE_NONE && ttValue < probBeta)) {
        Move pm[256]; Move* e = genMoves(pos, pm, G_NOISY); int cntm = 0;
        for (Move* mp = pm; mp < e && cntm < 8; mp++) {
            Move m = *mp; if (m == excluded || !pos.isLegal(m) || !pos.seeGE(m, probBeta - staticEval)) continue; cntm++;
            ss->move = m; ss->cont = &contH[(size_t)(pos.board[mFrom(m)] * 64 + mTo(m)) * 768];
            pos.doMove(m); int v = -qsearch(-probBeta, -probBeta + 1, ply + 1, 0);
            if (v >= probBeta) v = -search(-probBeta, -probBeta + 1, depth - 4, ply + 1, !cutNode);
            pos.undoMove();
            if (gStop.load(std::memory_order_relaxed)) return 0;
            if (v >= probBeta) { TTable.store(key, m, valueToTT(v - (probBeta - beta), ply), staticEval, depth - 3, B_LOWER, ss->ttPv); return v - (probBeta - beta); }
        }
    }
    // ---- move loop ----
    Move counterMv = MOVE_NONE; { Move pm = (ss - 1)->move; if (pm != MOVE_NONE) counterMv = counter[(ss - 1)->currentPieceTo]; }
    MovePicker mp(*this, rootNode ? MOVE_NONE : ttMove, ss->killers[0], ss->killers[1], counterMv, inCheck, false, ss);
    int best = -VALUE_INF, moveCount = 0, rootIdx = pvIdx - 1; Move bestMove = MOVE_NONE;
    Move quiets[64]; int nq = 0; Move caps[32]; int nc = 0;
    int bound = B_UPPER;
    for (;;) {
        Move m;
        if (rootNode) { ++rootIdx; if (rootIdx >= (int)root.size()) break; m = root[rootIdx].m; }
        else { m = mp.next(); if (m == MOVE_NONE) break; }
        if (m == excluded) continue;
        moveCount++;
        bool cap = isCap(m), noisy = isNoisy(m); int pc = pos.board[mFrom(m)], to = mTo(m);
        bool gives = pos.givesCheck(m);
        int extension = 0, newDepth = depth - 1;
        int lmrDepth = std::max(0, newDepth - Reductions[std::min(depth, 63)][std::min(moveCount, 63)]);
        int hist = noisy ? 0 : mainHist(us, m) * 2 + (ss - 1)->cont[pc * 64 + to] * 2 + (ss - 2)->cont[pc * 64 + to] + (ss - 4)->cont[pc * 64 + to] / 2;
        if (!rootNode && best > -MATE_IN_MAX && hasNonPawn(us)) {
            if (moveCount >= (improving ? 3 + depth * depth : (3 + depth * depth) / 2) && !inCheck) mp.skipQuiets = true;
            if (!noisy && !gives) {
                if (mp.skipQuiets && !inCheck) continue;
                if (lmrDepth <= 5 && hist < -3600 * depth) continue;
                if (!inCheck && lmrDepth <= 7 && staticEval + 105 + 92 * lmrDepth <= alpha) { mp.skipQuiets = true; continue; }
                if (!pos.seeGE(m, -26 * lmrDepth * lmrDepth)) continue;
            } else {
                int vv = mFlag(m) == EP ? PieceVal[PAWN] : (pos.board[to] == NO_PIECE ? 0 : PieceVal[typeOf(pos.board[to])]);
                if (!gives && !inCheck && lmrDepth < 7 && staticEval + 180 + vv + 130 * lmrDepth <= alpha && !(mFlag(m) & 8)) continue;
                if (!pos.seeGE(m, -145 * depth)) continue;
            }
        }
        // singular extension
        if (!rootNode && m == ttMove && !excluded && depth >= 6 && found && std::abs(ttValue) < MATE_IN_MAX && (t.bound & B_LOWER) && t.depth >= depth - 3) {
            int sBeta = ttValue - (56 + 79 * (ss->ttPv && !pv)) * depth / 64, sDepth = (depth - 1) / 2;
            ss->excluded = m; int v = search(sBeta - 1, sBeta, sDepth, ply, cutNode); ss->excluded = MOVE_NONE;
            if (gStop.load(std::memory_order_relaxed)) return 0;
            if (v < sBeta) { extension = 1; if (!pv && v < sBeta - 20 && ss->doubleExt <= 8) { extension = 2; ss->doubleExt++; } }
            else if (sBeta >= beta) return sBeta;
            else if (ttValue >= beta) extension = -2;
            else if (cutNode) extension = -1;
        } else if (gives && pv && !rootNode && pos.seeGE(m, 0) && depth <= 8) extension = 1;
        else if (inCheck && depth <= 2) extension = 1;
        newDepth += extension;
        ss->move = m; ss->currentPieceTo = pc * 64 + to; ss->cont = &contH[(size_t)(pc * 64 + to) * 768];
        U64 nb = nodes;
        pos.doMove(m);
        int v;
        if (pv) pvLen[ply + 1] = ply + 1;
        if (depth >= 2 && moveCount > 1 + (rootNode ? 1 : 0) && (!noisy || !ss->ttPv || true)) {
            int r = Reductions[std::min(depth, 63)][std::min(moveCount, 63)];
            if (cutNode) r += 2; if (pv) r -= 1; if (improving) r -= 1; if (ss->ttPv) r -= 1; if (gives) r -= 1; if (inCheck) r -= 1;
            if (ttCapture && !noisy) r += 1;
            if (noisy) r -= 1; else r -= hist / 7500;
            if (mp.skipQuiets && false) r++;
            int d = clampi(newDepth - r, 1, newDepth + 1);
            v = -search(-alpha - 1, -alpha, d, ply + 1, true);
            if (v > alpha && d < newDepth) v = -search(-alpha - 1, -alpha, newDepth, ply + 1, !cutNode);
        } else if (!pv || moveCount > 1) v = -search(-alpha - 1, -alpha, newDepth, ply + 1, !cutNode);
        else v = alpha + 1;
        if (pv && (moveCount == 1 || v > alpha)) { pvLen[ply + 1] = ply + 1; v = -search(-beta, -alpha, newDepth, ply + 1, false); }
        pos.undoMove();
        if (gStop.load(std::memory_order_relaxed)) return 0;
        if (rootNode) {
            RootMove& rm = root[rootIdx]; rm.nodes += nodes - nb;
            if (moveCount == 1 || v > alpha) {
                rm.score = v; rm.seldepth = seldepth; rm.pv.assign(1, m);
                for (int i = 1; i < pvLen[1]; i++) rm.pv.push_back(pvTable[1][i]);
                if (moveCount > 1) bestMoveChanges++;
            } else rm.score = -VALUE_INF;
        }
        if (v > best) {
            best = v;
            if (v > alpha) {
                bestMove = m;
                if (pv) { pvTable[ply][ply] = m; for (int i = ply + 1; i < pvLen[ply + 1]; i++) pvTable[ply][i] = pvTable[ply + 1][i]; pvLen[ply] = std::max(ply + 1, pvLen[ply + 1]); }
                if (v >= beta) { bound = B_LOWER; break; }
                alpha = v; bound = B_EXACT;
            }
        }
        if (m != bestMove) { if (noisy) { if (nc < 32) caps[nc++] = m; } else if (nq < 64) quiets[nq++] = m; }
    }
    if (moveCount == 0) return excluded ? alpha : inCheck ? -VALUE_MATE + ply : drawScore(nodes);
    // ---- history updates ----
    if (best >= beta) {
        int bonus = std::min(150 * depth - 80, 1600), malus = std::min(180 * depth - 90, 1500);
        int bpc = pos.board[mFrom(bestMove)], bto = mTo(bestMove);
        if (!isNoisy(bestMove)) {
            if (ss->killers[0] != bestMove) { ss->killers[1] = ss->killers[0]; ss->killers[0] = bestMove; }
            if ((ss - 1)->move != MOVE_NONE) counter[(ss - 1)->currentPieceTo] = bestMove;
            gravity(mainHist(us, bestMove), bonus);
            for (int k : {1, 2, 4}) { SS* p2 = ss - k; if (p2->move != MOVE_NONE || k > 0) gravity(p2->cont[bpc * 64 + bto], bonus * (k == 4 ? 1 : 2) / 2 + (k == 1 ? bonus / 2 : 0)); }
            for (int i = 0; i < nq; i++) { Move q = quiets[i]; int qp = pos.board[mFrom(q)], qt = mTo(q);
                gravity(mainHist(us, q), -malus); for (int k : {1, 2, 4}) gravity((ss - k)->cont[qp * 64 + qt], -malus / 2); }
        } else {
            int vt = mFlag(bestMove) == EP ? PAWN : typeOf(pos.board[bto]); gravity(capHist(bpc, bto, vt), bonus);
        }
        for (int i = 0; i < nc; i++) { Move c = caps[i]; int cp = pos.board[mFrom(c)], ct = mTo(c); int vt = mFlag(c) == EP ? PAWN : (pos.board[ct] == NO_PIECE ? PAWN : typeOf(pos.board[ct])); gravity(capHist(cp, ct, vt), -malus); }
    }
    if (!excluded && !(rootNode && pvIdx > 0)) TTable.store(key, bestMove, valueToTT(best, ply), staticEval == VALUE_NONE ? VALUE_NONE : staticEval - correction(pos), depth, bound, ss->ttPv);
    if (!inCheck && !excluded && staticEval != VALUE_NONE && (!bestMove || !isNoisy(bestMove)) &&
        !(bound == B_LOWER && best <= staticEval) && !(bound == B_UPPER && best >= staticEval) && std::abs(best) < MATE_IN_MAX)
        updateCorrection(pos, best - staticEval, depth);
    return best;
}

// ------------------------- Time management & driver -------------------------
static void computeTimes(int us, int gamePly) {
    const Limits& L = gLim; gOptimum = gMaximum = 0;
    if (L.movetime > 0) { gOptimum = gMaximum = std::max<int64_t>(1, L.movetime - gOverhead / 2); return; }
    if (L.time[us] > 0) {
        int64_t tl = L.time[us], inc = L.inc[us]; int mtg = L.movestogo > 0 ? std::min(L.movestogo, 40) : 36;
        // spend more early in the game; keep a reserve
        double moveNum = gamePly / 2.0; double f = L.movestogo > 0 ? 1.0 : (moveNum < 12 ? 1.1 : moveNum < 40 ? 1.0 : 0.9);
        int64_t avail = std::max<int64_t>(1, tl - gOverhead * 2);
        int64_t base = (int64_t)((double)avail / mtg * f) + inc * 3 / 4;
        gOptimum = std::min<int64_t>(base, avail * 6 / 10);
        gMaximum = std::min<int64_t>((int64_t)(base * 4.2), avail * 7 / 10); if (gMaximum < gOptimum) gMaximum = gOptimum;
        gOptimum = std::max<int64_t>(gOptimum, 1); gMaximum = std::max<int64_t>(gMaximum, 2);
    }
}

static std::vector<std::unique_ptr<Thread>> gThreadPool;
static Position gRoot; static Move gBestMove = MOVE_NONE, gPonder = MOVE_NONE; static int gLastScore = 0, gLastDepth = 0;
static std::mutex gIoMutex;

static std::string scoreStr(int v) {
    if (v >= MATE_IN_MAX) return "mate " + std::to_string((VALUE_MATE - v + 1) / 2);
    if (v <= -MATE_IN_MAX) return "mate " + std::to_string(-(VALUE_MATE + v) / 2);
    return "cp " + std::to_string(v);
}
static U64 totalNodes() { U64 n = 0; for (auto& t : gThreadPool) n += t->nodes; return n; }

static void printInfo(Thread& th, int depth, int multi) {
    int64_t ms = std::max<int64_t>(1, elapsedMs()); U64 n = totalNodes();
    for (int i = 0; i < multi && i < (int)th.root.size(); i++) {
        RootMove& rm = th.root[i]; int sc = rm.score == -VALUE_INF ? rm.prev : rm.score; if (sc == -VALUE_INF) continue;
        std::string s = "info depth " + std::to_string(depth) + " seldepth " + std::to_string(std::max(rm.seldepth, depth)) + " multipv " + std::to_string(i + 1) + " score " + scoreStr(sc) +
                        " nodes " + std::to_string(n) + " nps " + std::to_string(n * 1000 / ms) + " hashfull " + std::to_string(TTable.hashfull()) + " time " + std::to_string(ms) + " pv";
        for (Move m : rm.pv) s += " " + moveStr(m);
        std::lock_guard<std::mutex> lk(gIoMutex); puts(s.c_str()); fflush(stdout);
    }
}

void Thread::idLoop() {
    static const int skipSize[20] = {1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4}, skipPhase[20] = {0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7};
    int maxDepth = gLim.depth > 0 ? std::min(gLim.depth, MAX_PLY - 2) : MAX_PLY - 2;
    int multi = std::min<int>(gMultiPV, (int)root.size());
    for (int i = 0; i < 8; i++) { stack[i] = SS(); stack[i].cont = &contH[(size_t)(12 * 64) * 768]; stack[i].staticEval = VALUE_NONE; stack[i].move = MOVE_NONE; }
    for (int i = 8; i < MAX_PLY + 16; i++) { stack[i].killers[0] = stack[i].killers[1] = 0; stack[i].cont = &contH[(size_t)(12 * 64) * 768]; stack[i].excluded = 0; stack[i].move = MOVE_NONE; stack[i].staticEval = VALUE_NONE; stack[i].inCheck = false; }
    Move lastBest = MOVE_NONE; int stableCount = 0, prevScore = 0; double totalEff = 1;
    for (rootDepth = 1; rootDepth <= maxDepth; rootDepth++) {
        if (id > 0) { int i = (id - 1) % 20; if (((rootDepth + skipPhase[i]) / skipSize[i]) % 2) continue; }
        for (auto& rm : root) { rm.prev = rm.score; rm.score = -VALUE_INF; rm.nodes = 0; }
        seldepth = 0; bestMoveChanges = 0;
        for (pvIdx = 0; pvIdx < multi && !gStop.load(); pvIdx++) {
            int pr = root[pvIdx].prev; if (pr == -VALUE_INF) pr = 0;
            int delta = 10 + pr * pr / 12000, a = -VALUE_INF, b = VALUE_INF;
            if (rootDepth >= 5) { a = std::max(pr - delta, -VALUE_INF); b = std::min(pr + delta, VALUE_INF); }
            int adj = rootDepth;
            for (;;) {
                int sc = search(a, b, adj, 0, false);
                std::stable_sort(root.begin() + pvIdx, root.end(), [](const RootMove& x, const RootMove& y) { return x.score > y.score; });
                if (gStop.load()) break;
                if (sc <= a) { b = (a + b) / 2; a = std::max(sc - delta, -VALUE_INF); adj = rootDepth; if (id == 0 && !gQuiet && rootDepth > 8 && elapsedMs() > 3000) printInfo(*this, rootDepth, 1); }
                else if (sc >= b) { b = std::min(sc + delta, VALUE_INF); if (adj > 1 && std::abs(sc) < MATE_IN_MAX) adj--; }
                else break;
                delta += delta / 3 + 4;
                if (delta > 3000) { a = -VALUE_INF; b = VALUE_INF; }
            }
            std::stable_sort(root.begin(), root.begin() + pvIdx + 1, [](const RootMove& x, const RootMove& y) { return x.score > y.score; });
        }
        bool done = !gStop.load();
        if (root[0].score > -VALUE_INF && (done || root[0].score > -VALUE_INF)) { if (id == 0 || true) { /* result valid */ } }
        if (id == 0) {
            if (root[0].score > -VALUE_INF) { gBestMove = root[0].m; gPonder = root[0].pv.size() > 1 ? root[0].pv[1] : MOVE_NONE; gLastScore = root[0].score; gLastDepth = rootDepth; }
            if (done && !gQuiet) printInfo(*this, rootDepth, multi);
        }
        if (!done) break;
        completed = rootDepth;
        if (id == 0) {
            int sc = root[0].score;
            if (root[0].m == lastBest) stableCount++; else { stableCount = 0; lastBest = root[0].m; }
            bool timed = !gLim.infinite && (gLim.time[gRoot.side] > 0 || gLim.movetime > 0);
            if (timed && rootDepth >= 1) {
                if (root.size() == 1 && !gLim.movetime && rootDepth >= 2) { gStop = true; break; }
                if (std::abs(sc) >= MATE_IN_MAX && VALUE_MATE - std::abs(sc) <= rootDepth && rootDepth >= 6 && !gLim.movetime) { gStop = true; break; }
                if (gLim.time[gRoot.side] > 0 && !gLim.movetime) {
                    double fall = clampi(80 + 3 * (prevScore - sc) + 6 * 0, 70, 160) / 100.0; if (rootDepth < 6) fall = 1.0;
                    double inst = 1.0 + 0.55 * std::min(bestMoveChanges, 4) / 4.0; double stab = stableCount >= 6 ? 0.72 : stableCount >= 3 ? 0.88 : 1.0;
                    double eff = 1.0; U64 tn = nodes ? nodes : 1; if (rootDepth >= 10) { double fr = (double)root[0].nodes / (double)tn; eff = fr > 0.85 ? 0.7 : fr > 0.7 ? 0.85 : 1.0; }
                    totalEff = fall * inst * stab * eff;
                    if (elapsedMs() >= (int64_t)(gOptimum * totalEff)) { gStop = true; break; }
                } else if (gLim.movetime > 0 && elapsedMs() >= gMaximum * 55 / 100 && rootDepth >= 8) {
                    // movetime: finish when next iteration cannot plausibly complete
                    if (elapsedMs() * 2 > gMaximum) { /* soft: let hard limit decide */ }
                }
            }
            prevScore = sc;
        }
    }
    if (id == 0 && gLim.infinite) while (!gStop.load()) std::this_thread::sleep_for(std::chrono::milliseconds(2));
}

static void runSearch(Limits L) {
    gLim = L; gStop = false; gStart = std::chrono::steady_clock::now(); TTable.newSearch();
    int us = gRoot.side; computeTimes(us, (gRoot.fullmove - 1) * 2 + (us == BLACK)); 
    while ((int)gThreadPool.size() < gThreads) { gThreadPool.emplace_back(new Thread()); gThreadPool.back()->alloc(); gThreadPool.back()->id = (int)gThreadPool.size() - 1; }
    while ((int)gThreadPool.size() > gThreads) gThreadPool.pop_back();
    Move legal[256]; int nl = genLegal(gRoot, legal);
    gBestMove = MOVE_NONE; gPonder = MOVE_NONE;
    if (nl == 0) { printf("info depth 0 score %s\n", gRoot.inCheck() ? "mate 0" : "cp 0"); printf("bestmove 0000\n"); fflush(stdout); return; }
    std::vector<RootMove> rm;
    for (int i = 0; i < nl; i++) { bool ok = L.searchmoves.empty(); for (Move m : L.searchmoves) if (m == legal[i]) ok = true; if (ok) { RootMove r; r.m = legal[i]; rm.push_back(r); } }
    if (rm.empty()) for (int i = 0; i < nl; i++) { RootMove r; r.m = legal[i]; rm.push_back(r); }
    // order root moves by a quick static ordering (captures first) to help depth 1
    for (auto& t : gThreadPool) { t->pos = gRoot; t->nodes = 0; t->root = rm; t->seldepth = 0; t->nmpMinPly = 0; t->completed = 0; }
    gBestMove = rm[0].m;
    std::vector<std::thread> helpers;
#ifndef __EMSCRIPTEN__
    for (int i = 1; i < (int)gThreadPool.size(); i++) helpers.emplace_back([i]() { gThreadPool[i]->idLoop(); });
#endif
    gThreadPool[0]->idLoop();
    gStop = true; for (auto& h : helpers) h.join();
    std::string s = "bestmove " + moveStr(gBestMove); if (gPonder) s += " ponder " + moveStr(gPonder);
    { std::lock_guard<std::mutex> lk(gIoMutex); puts(s.c_str()); fflush(stdout); }
}
// ----------------------------------- UCI ------------------------------------
static const char* START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
static std::thread gSearchThread; static bool gSearching = false;
static void waitSearch() { if (gSearchThread.joinable()) gSearchThread.join(); gSearching = false; }
static std::string gLastFenForReset;

static void setPosition(std::istringstream& is) {
    std::string tok, fen; is >> tok;
    if (tok == "startpos") { fen = START_FEN; is >> tok; }
    else if (tok == "fen") { while (is >> tok && tok != "moves") fen += tok + " "; }
    else return;
    Position np; if (!np.setFen(fen)) { printf("info string illegal fen\n"); fflush(stdout); return; }
    if (tok == "moves") {
        std::string ms;
        while (is >> ms) {
            Move m = parseMove(np, ms);
            if (m == MOVE_NONE) { printf("info string illegal move %s\n", ms.c_str()); fflush(stdout); break; }
            if (np.sp > STACK_SZ - 300) { std::string f = np.fen(); Position t; t.setFen(f); np = t; }
            np.doMove(m);
        }
    }
    gRoot = np;
}
static std::string gameStatus(Position& p) {
    Move l[256]; int n = genLegal(p, l);
    if (n == 0) return p.inCheck() ? "checkmate" : "stalemate";
    if (p.insufficient()) return "draw_material";
    if (p.st[p.sp].halfmove >= 100) return "draw_fifty";
    int cnt = 0; const StateInfo& s = p.st[p.sp]; for (int i = 2; i <= std::min(s.halfmove, p.sp); i += 2) if (p.st[p.sp - i].key == s.key) cnt++;
    if (cnt >= 2) return "draw_repetition";
    return "ongoing";
}
static const char* BENCH_FENS[] = {
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/R3KB1R w KQ - 0 9",
    "2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - - 0 1",
    "8/8/4kpp1/3p1b2/p6P/2B5/6P1/6K1 b - - 0 1",
    "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1"};

static void handleGo(std::istringstream& is) {
    Limits L; std::string t;
    while (is >> t) {
        if (t == "wtime") { is >> L.time[0]; } else if (t == "btime") { is >> L.time[1]; } else if (t == "winc") { is >> L.inc[0]; } else if (t == "binc") { is >> L.inc[1]; }
        else if (t == "movestogo") is >> L.movestogo; else if (t == "movetime") is >> L.movetime; else if (t == "depth") is >> L.depth;
        else if (t == "nodes") is >> L.nodes; else if (t == "infinite") L.infinite = true; else if (t == "mate") is >> L.mate;
        else if (t == "searchmoves") { std::streampos pos = is.tellg(); std::string mv; Position tmp = gRoot;
            while (is >> mv) { Move m = parseMove(tmp, mv); if (m == MOVE_NONE) { is.clear(); is.seekg(pos); break; } L.searchmoves.push_back(m); pos = is.tellg(); } }
    }
    if (!L.depth && !L.nodes && !L.movetime && !L.time[0] && !L.time[1] && !L.infinite) L.movetime = 1000;
#ifdef __EMSCRIPTEN__
    if (L.infinite) { L.infinite = false; L.movetime = 2000; }
    runSearch(L);
#else
    waitSearch(); gSearching = true;
    gSearchThread = std::thread([L]() { runSearch(L); });
#endif
}
static void uciCmd(const std::string& line, bool& quit) {
    std::istringstream is(line); std::string cmd; is >> cmd;
    if (cmd == "uci") {
        printf("id name Atelier 4\nid author Atelier\n");
        printf("option name Hash type spin default 128 min 1 max 16384\noption name Threads type spin default 1 min 1 max 64\noption name MultiPV type spin default 1 min 1 max 64\n");
        printf("option name Move Overhead type spin default 30 min 0 max 5000\noption name Clear Hash type button\nuciok\n"); fflush(stdout);
    } else if (cmd == "isready") { printf("readyok\n"); fflush(stdout); }
    else if (cmd == "ucinewgame") { waitSearch(); TTable.clear(); for (auto& t : gThreadPool) t->clearHist(); }
    else if (cmd == "position") { waitSearch(); setPosition(is); }
    else if (cmd == "go") handleGo(is);
    else if (cmd == "stop") { gStop = true; waitSearch(); }
    else if (cmd == "quit") { gStop = true; waitSearch(); quit = true; }
    else if (cmd == "setoption") {
        std::string tok, name, val; is >> tok; while (is >> tok && tok != "value") name += (name.empty() ? "" : " ") + tok; while (is >> tok) val += (val.empty() ? "" : " ") + tok;
        waitSearch();
        int v = atoi(val.c_str());
        if (name == "Hash") TTable.resize(std::max(1, std::min(v, 16384)));
        else if (name == "Threads") gThreads = std::max(1, std::min(v, 64));
        else if (name == "MultiPV") gMultiPV = std::max(1, std::min(v, 64));
        else if (name == "Move Overhead") gOverhead = std::max(0, std::min(v, 5000));
        else if (name == "Clear Hash") { TTable.clear(); for (auto& t : gThreadPool) t->clearHist(); }
    }
    else if (cmd == "d") { printf("%s\nkey %llx\n", gRoot.fen().c_str(), (unsigned long long)gRoot.key()); fflush(stdout); }
    else if (cmd == "eval") { EVDBG = true; int e = evaluate(gRoot); EVDBG = false; printf("eval %d (side to move)\n", e); fflush(stdout); }
    else if (cmd == "status") { printf("status %s\n", gameStatus(gRoot).c_str()); fflush(stdout); }
    else if (cmd == "legal") { Move l[256]; int n = genLegal(gRoot, l); printf("legal"); for (int i = 0; i < n; i++) printf(" %s", moveStr(l[i]).c_str()); printf("\n"); fflush(stdout); }
    else if (cmd == "perft") { int d = 1; is >> d; Move l[256]; int n = genLegal(gRoot, l); U64 tot = 0; for (int i = 0; i < n; i++) { gRoot.doMove(l[i]); U64 c = d > 1 ? perft(gRoot, d - 1) : 1; gRoot.undoMove(); tot += c; printf("%s: %llu\n", moveStr(l[i]).c_str(), (unsigned long long)c); } printf("\nNodes searched: %llu\n", (unsigned long long)tot); fflush(stdout); }
    else if (cmd == "bench") {
        int d = 12; is >> d; U64 tn = 0; auto t0 = std::chrono::steady_clock::now(); bool q = gQuiet; gQuiet = true;
        for (auto f : BENCH_FENS) { Position p; p.setFen(f); gRoot = p; TTable.clear(); for (auto& t : gThreadPool) t->clearHist(); Limits L; L.depth = d; runSearch(L); tn += totalNodes(); }
        gQuiet = q; double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        printf("bench: %llu nodes %.0f ms %.0f nps\n", (unsigned long long)tn, ms, tn * 1000.0 / ms); fflush(stdout);
    }
}
static void globalInit() {
    static bool done = false; if (done) return; done = true;
    initTables(); initEval(); initKPK(); initSearch(); TTable.resize(128); gRoot.setFen(START_FEN);
}
#ifdef __EMSCRIPTEN__
extern "C" EMSCRIPTEN_KEEPALIVE void atelier_cmd(const char* s) { globalInit(); bool q = false; uciCmd(s, q); }
int main() { globalInit(); return 0; }
#else
int main(int argc, char** argv) {
    globalInit(); setvbuf(stdout, nullptr, _IOLBF, 0);
    bool quit = false;
    if (argc > 1) { std::string c; for (int i = 1; i < argc; i++) c += std::string(argv[i]) + " "; uciCmd(c, quit); waitSearch(); return 0; }
    std::string line; while (!quit && std::getline(std::cin, line)) { if (!line.empty()) uciCmd(line, quit); }
    gStop = true; waitSearch(); return 0;
}
#endif
