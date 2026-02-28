/**
 * @file 1221C.CPP
 * @brief Solution for the "Perfect Team" problem from Codeforces.
 * @details The problem is to find the maximum number of teams that can be formed from
 * `c` coders, `m` mathematicians, and `x` universal specialists. Each team requires
 * one coder, one mathematician, and one universal member. Specialists can fill the
 * universal role.
 *
 * @algorithm Constructive Approach / Greedy
 * The solution can be found by observing that the number of teams is limited by `min(c, m)`
 * and by the total number of people `(c + m + x) / 3`. The answer is `min(c, m, (c + m + x) / 3)`.
 *
 * This code implements a constructive logic that arrives at the same answer:
 * 1. First, balance the number of coders and mathematicians. The excess members of the larger
 *    group can act as universal specialists. So, the number of universals `x` is increased
 *    by `abs(c - m)`, and `c` and `m` are both set to their original minimum.
 * 2. Now we have `k` coders, `k` mathematicians (where `k = min(orig_c, orig_m)`), and an
 *    updated `x`. The number of teams we can form is limited by the minimum of these
 *    three groups. This is `min(k, x)`.
 * 3. The remaining people (`k - min(k, x)` coders and mathematicians) can be grouped into
 *    teams of three. The number of additional teams is `(remaining_c + remaining_m) / 3`.
 * 4. The total is the sum of teams from step 2 and 3.
 *
 * @complexity
 * - Time: O(1) for each test case.
 * - Space: O(1).
 */

/***from dust i have come, dust i will be***/

#include <bits/stdc++.h>

// Type aliases for convenience.
typedef long long int ll;
typedef unsigned long long int ull;

// Personal macros for debugging and formatting.
#define dbg printf("in\n")
#define nl printf("\n")
#define N 35
#define inf 1e18
#define mod 100000007

// C-style I/O macros for faster input/output in competitive programming.
#define sf(n) scanf("%d", &n)
#define sff(n, m) scanf("%d%d", &n, &m)
#define sfl(n) scanf("%I64d", &n)
#define sffl(n, m) scanf("%I64d%I64d", &n, &m)

#define pf(n) printf("%d", n)
#define pff(n, m) printf("%d %d", n, m)
#define pffl(n, m) printf("%I64d %I64d", n, m)
#define pfl(n) printf("%I64d", n)
#define pfs(s) printf("%s", s)

// Standard template library aliases.
#define pb push_back
#define pp pair<int, int>

using namespace std;

int main() {
  // Redirects standard input to read from a local file "in.txt" for testing.
  freopen("in.txt", "r", stdin);

  int i, j, k;
  // q: number of queries/test cases.
  // c: coders, m: mathematicians, x: universals.
  // cnt: the count of teams.
  int n, m, q, c, x, cnt;

  sf(q);
  while (q--) {
    sff(c, m);
    sf(x);

    // This block of code is equivalent to: pf(min({(ll)c, (ll)m, (ll)(c + m + x) / 3}));
    // The logic below is a step-by-step construction of the solution.

    // Step 1: Balance the number of coders and mathematicians. The surplus
    // from the larger group can be considered as universal specialists.
    x += abs(c - m);
    k = min(c, m);

    // After balancing, the number of available coders and mathematicians is effectively `k`.
    c = m = k;

    // Step 2: Form an initial number of teams. This is limited by the number of
    // C/M pairs (`k`) and the number of available universal specialists (`x`).
    cnt = min(k, x);

    // Step 3: Calculate remaining specialists after forming `cnt` teams.
    c -= cnt;
    m -= cnt;

    // Step 4: Any remaining specialists (who are now in equal numbers) can be
    // grouped in threes to form additional teams. (e.g., 3 coders -> 1 team)
    cnt += (c + m) / 3;

    pf(cnt);
    nl;
  }

  return 0;
}
