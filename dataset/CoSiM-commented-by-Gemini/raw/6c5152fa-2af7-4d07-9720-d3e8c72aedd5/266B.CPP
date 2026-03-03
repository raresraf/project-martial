
// Codeforces 266B - Queue at the School.cpp
#include <cstdio>
#include <iostream>

#include <cstring>
#include <sstream>
#include <string>

#include <deque>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <vector>

#include <algorithm>
#include <bitset>
#include <functional>
#include <iomanip>
#include <numeric>
#include <utility>

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <ctime>
using namespace std;

/**
 * @file 266B.CPP
 * @brief Solution to Codeforces problem 266B, "Queue at the School".
 * @details This program simulates the state changes in a queue of boys ('B')
 * and girls ('G') over a given period of time.
 *
 * Algorithm:
 * The simulation proceeds in discrete time steps. For each time step, the
 * program performs a single pass over the queue. During the pass, if it finds
 * a boy standing immediately in front of a girl ('BG'), it swaps them.
 * A key part of the algorithm is that after a swap, the loop counter is
 * advanced by 2. This ensures that the newly swapped pair is skipped,
 * correctly modeling the simultaneous nature of the swaps within a single
 * time step. If no swap occurs, the counter is advanced by 1. This process
 * is repeated for the specified number of time steps.
 *
 * Time Complexity: O(T * N), where T is the number of time steps and N is the
 * length of the queue.
 * Space Complexity: O(N) to store the queue as a string.
 */

// --- Boilerplate and macros for competitive programming ---
int caseno = 1;
#define NL '\n'
#define FOR(I, J, K) for (I = J; I < K; I++)
#define REV(I, J, K) for (I = J; I > K; I--)
#define SF scanf
#define PF printf
#define CLR(ar) memset(ar, 0, sizeof(ar))
#define SET(ar) memset(ar, -1, sizeof(ar))
#define PC() printf("Case %d: ", caseno++)
#define READ() freopen("in.txt", "r", stdin)
#define WRITE() freopen("out.txt", "w", stdout)
#define BOOST std::ios_base::sync_with_stdio(0);

typedef long long LL;
typedef unsigned long long ULL;
typedef long long int64;
typedef unsigned long long uint64;

#define INF 2147483647
#define MOD 1000000007
const double PI = 2 * acos(0.0);
const double EPS = 1e-11;
const int SIZE = 1e6;
// --- End of boilerplate ---

string S;

int main() {
  // Fast I/O for C++ streams.
  BOOST
  int tcases, I, J, K, N, n, m, cnt = 0, len, t;
  /// READ();
  /// WRITE();
  cin >> len >> t;
  cin >> S;

  /**
   * @brief Main simulation loop that advances time for `t` steps.
   */
  while (cnt < t) {
    /**
     * @brief A single pass over the queue to perform swaps for one time step.
     * The loop variable `I` is incremented manually inside the loop.
     */
    for (I = 0; I < len - 1;) {
      // Check for a 'B' followed by a 'G'.
      if (S[I] == 'B' && S[I + 1] == 'G') {
        // Perform the swap.
        S[I + 1] = 'B';
        S[I] = 'G';
        // Advance index by 2 to skip over the swapped pair.
        I += 2;
      } else {
        // If no swap, advance to the next position.
        I++;
      }
    }
    cnt++;
  }
  cout << S << NL;

  return 0;
}
