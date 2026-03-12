/***from dust i have come, dust i will be***/

/**
 * @5e6d8815-e5b5-40a7-a590-283f9ba081c6/1221C.CPP
 * @brief This program solves Codeforces problem 1221C, "Perfect Team".
 * It determines the maximum number of complete teams that can be formed
 * given `c` programmers, `m` mathematicians, and `x` generalists.
 * A team requires one programmer, one mathematician, and one generalist.
 * The solution incorporates a greedy strategy where surplus specialists
 * can contribute to the generalist count.
 *
 * Algorithm: Greedy calculation involving balancing specialist counts and combining with generalists.
 * Time Complexity: O(T) where T is the number of test cases.
 * Space Complexity: O(1) for storing variables.
 */

#include <bits/stdc++.h> // Includes many standard C++ library headers.

typedef long long int ll;     // Defines 'll' as a shorthand for long long int.
typedef unsigned long long int ull; // Defines 'ull' as a shorthand for unsigned long long int.

#define dbg printf("in\n") // Debugging macro for printing "in".
#define nl printf("\n")    // Macro for printing a newline.
#define N 35               // Defines a constant N (unused in this specific solution).
#define inf 1e18           // Defines 'inf' as a large number for infinity.
#define mod 100000007      // Defines 'mod' for a common modulo value.

#define sf(n) scanf("%d", &n)       // Macro for scanning an integer.
#define sff(n, m) scanf("%d%d", &n, &m) // Macro for scanning two integers.
#define sfl(n) scanf("%I64d", &n)     // Macro for scanning a long long int (Microsoft specific).
#define sffl(n, m) scanf("%I64d%I64d", &n, &m) // Macro for scanning two long long ints.

#define pf(n) printf("%d", n)         // Macro for printing an integer.
#define pff(n, m) printf("%d %d", n, m) // Macro for printing two integers.
#define pffl(n, m) printf("%I64d %I64d", n, m) // Macro for printing two long long ints.
#define pfl(n) printf("%I64d", n)     // Macro for printing a long long int.
#define pfs(s) printf("%s", s)        // Macro for printing a string.

#define pb push_back     // Defines 'pb' as a shorthand for vector::push_back.
#define pp pair<int, int> // Defines 'pp' as a shorthand for pair<int, int>.

using namespace std; // Uses the standard namespace.

int main() {
  freopen("in.txt", "r", stdin); // Redirects standard input from "in.txt" (for local testing).

  int i, j, k; // Declares integer variables: i, j (unused), k (temporary team count).
  int n, m, q, c, x, cnt; // Declares more integer variables: n, m (used as counts), q (test cases), c, x (counts), cnt (final team count).

  sf(q); // Reads the number of test cases 'q' from standard input.
  /**
   * Block Logic: Processes each test case to calculate the maximum number of teams.
   * Invariant: For each test case, 'cnt' stores the maximum number of teams that can be formed.
   */
  while (q--) { // Loops 'q' times, once for each test case.
    sff(c, m); // Reads 'c' (programmers) and 'm' (mathematicians).
    sf(x);     // Reads 'x' (generalists).

    // Block Logic: This crucial step transforms surplus specialists into generalists.
    // The absolute difference between programmers and mathematicians (`abs(c-m)`)
    // represents the number of specialists that are not paired with their counterpart.
    // These surplus specialists can be used as generalists.
    x += abs(c - m); 
    
    // Block Logic: Balances the number of programmers and mathematicians.
    // 'k' will hold the maximum number of teams that can be formed if only
    // programmers and mathematicians were considered.
    k = min(c, m); 

    // Inline: Adjusts 'c' and 'm' to this balanced count 'k'.
    // Any excess beyond 'k' has effectively been transferred to 'x'.
    c = m = k; 

    // Block Logic: First pass for forming teams. The number of teams `cnt` is initially
    // limited by the minimum of balanced specialists (`k`) and available generalists (`x`).
    cnt = min(k, x); 

    // Inline: Subtracts the number of teams formed from 'c' and 'm'.
    c -= cnt; 
    m -= cnt; 
    
    // Block Logic: Second pass for forming teams from remaining specialists.
    // After the initial `cnt` teams are formed, any remaining 'c' and 'm'
    // are now treated solely as potential generalists. Each team needs one generalist.
    // The sum of remaining 'c' and 'm' is effectively added to the pool of generalists.
    // These are then divided by 3 (since each team needs one generalist, and 'c' and 'm'
    // contribute to the generalist pool now) to form additional teams.
    cnt += (c + m) / 3; 

    pf(cnt); // Prints the final calculated count of teams.
    nl;      // Prints a newline.
  }

  return 0; // Indicates successful program termination.
}
