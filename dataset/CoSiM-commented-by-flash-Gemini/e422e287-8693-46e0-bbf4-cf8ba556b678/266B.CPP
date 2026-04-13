/**
 * @file 266B.CPP
 * @brief This program simulates a queue rearrangement process over a specified
 *        number of time steps. In each step, adjacent pairs of 'BG' (Boy-Girl)
 *        are swapped to 'GB' (Girl-Boy). The simulation models students
 *        in a queue, with boys moving forward and girls moving backward
 *        if a specific pattern is met.
 * Algorithm: Simulation, String manipulation.
 * Time Complexity: O(T * N) where T is the number of time steps and N is the
 *                  length of the queue string, as each time step involves a
 *                  linear scan of the queue.
 * Space Complexity: O(N) for storing the queue string.
 */
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

const double PI = 3.1415926535897932384626433832795;

int main() {
  // n: length of the queue.
  // t: number of time steps for the simulation.
  int n, t;
  // Reads the length of the queue and the number of time steps.
  cin >> n >> t;
  string s; // String representing the queue (e.g., "BGGB").
  // Reads the initial configuration of the queue.
  cin >> s;

  /**
   * Block Logic: Outer loop iterates for 't' time steps, simulating the queue rearrangement.
   * Invariant: After each iteration, one time step of swaps has been completed throughout the queue.
   */
  REP(i, t) {
    /**
     * Block Logic: Inner loop traverses the queue from left to right, performing 'BG' to 'GB' swaps.
     * Invariant: `j` increments, and if `s[j]` is 'B' and `s[j+1]` is 'G', they are swapped,
     *            and `j` is incremented an extra time to avoid re-checking the newly swapped 'G'.
     */
    REP(j, n - 1) { // Iterate up to n-1 to safely access s[j+1].
      if (s[j] == 'B' && s[j + 1] == 'G') { // Checks for a 'BG' pattern.
        s[j] = 'G';     // Swap 'B' to 'G'.
        s[j + 1] = 'B'; // Swap 'G' to 'B'.
        j++;            // Skip the next position as 'G' just moved there.
      }
    }
  }
  // Prints the final state of the queue after all time steps.
  cout << s << endl;
  return 0;
}
