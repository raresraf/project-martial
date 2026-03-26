/**
 * @file 266B.CPP
 * @brief Solution to Codeforces problem 266B - Queue at the School.
 * @description This program simulates the arrangement of a queue of boys ('B') and
 *              girls ('G') over time. In each second, any boy standing immediately
 *              in front of a girl swaps places with her. The program calculates
 *              the final state of the queue after 't' seconds.
 *
 * Algorithm: The simulation proceeds in discrete time steps.
 * 1. An outer loop runs for 't' iterations, representing each second.
 * 2. An inner loop scans the queue from left to right.
 * 3. Whenever a 'B' is found immediately before a 'G', they are swapped.
 * 4. After a swap at index `j`, the inner loop's index is incremented (`j++`).
 *    This is a critical detail that ensures a boy who has just swapped does not
 *    participate in another swap within the same time step, correctly modeling
 *    the simultaneous swaps occurring in one second.
 *
 * Time Complexity: O(N * T), where N is the number of children and T is the time in seconds.
 * Space Complexity: O(N) to store the queue string.
 */

// --- Boilerplate and Macros for Competitive Programming ---
#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#define REP(i, n) for (int i = 0; i < (int)(n); ++i)
#define FOR(i, m, n) for (int i = (m); i < (int)(n); ++i)
#define ALL(x) (x).begin(), (x).end()
#define INF 2000000000
#ifdef LOCAL
#define dprintf(...) fprintf(stdout, __VA_ARGS__)
#else
#define dprintf(...) 0
#endif

using namespace std;

typedef long long LL;
typedef unsigned long long ULL;
typedef unsigned int uint;
// --- End of Boilerplate ---

const double PI = 3.1415926535897932384626433832795;

int main() {
  int n, t; // n: number of children, t: time in seconds
  cin >> n >> t;
  string s; // The queue arrangement
  cin >> s;

  /**
   * Block Logic (Outer Loop): Simulates the passage of 't' seconds.
   */
  REP(i, t) {
    /**
     * Block Logic (Inner Loop): Scans the queue for possible swaps in a single time step.
     */
    REP(j, n - 1) {
      // Pre-condition: Check for a boy ('B') followed immediately by a girl ('G').
      if (s[j] == 'B' && s[j + 1] == 'G') {
        // Perform the swap.
        s[j] = 'G';
        s[j + 1] = 'B';
        // Inline: Crucially, increment 'j' to skip the next position.
        // This ensures that the boy who just moved does not move again in the same time step,
        // correctly modeling simultaneous swaps.
        j++;
      }
    }
  }
  
  // Output the final state of the queue.
  cout << s << endl;
  return 0;
}
