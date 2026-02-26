/**
 * @file 266B.CPP
 * @brief Solves the "Queue at the School" problem from Codeforces.
 * @author Jos Luiz da Silva Neto
 *
 * @details This program simulates the arrangement of a queue of boys ('B') and
 * girls ('G'). Over `t` seconds, any boy standing immediately before a girl
 * will swap places with her. The program determines the final state of the queue.
 *
 * @algorithm
 * A direct, discrete-time simulation is performed.
 * 1. The simulation runs for `t` time steps (seconds), controlled by an outer loop.
 * 2. In each step, an inner loop scans the queue from left to right.
 * 3. If a 'B' is found at position `i` and a 'G' at `i+1`, they are swapped.
 * 4. To correctly model that all swaps in a single second happen
 *    "simultaneously," the loop counter `i` is incremented an extra time after a
 *    swap. This prevents the same characters from being involved in multiple
 *    swaps within the same time step.
 *
 * @complexity
 * Time Complexity: O(T * N), where T is the time in seconds and N is the length
 * of the queue.
 * Space Complexity: O(N) to store the queue as a string.
 */

/// Jos Luiz da Silva Neto
/// Computer Engineering - Federal University of Itajub
#include <bits/stdc++.h>

using namespace std;

// A collection of personal macros and typedefs for competitive programming.
// Most are not used in this specific solution.
#define INF 0x3F3F3F3F
#define LINF 0x3F3F3F3F3F3F3F3FLL
#define pb push_back
#define mp make_pair
#define F first
#define S second

typedef pair<int, int> pii;
typedef vector<int> vi;
typedef vector<pii> vii;
typedef long long ll;
typedef unsigned long long ull;

#define D(x) cout << #x << " = " << x << endl;

// Unused arrays for grid-based problems (e.g., BFS/DFS).
int vx[] = {1, 0, -1, 0};
int vy[] = {0, 1, 0, -1};

int main() {
  int n, t; // n: number of children, t: time in seconds
  cin >> n >> t;
  string x; // Represents the queue
  cin >> x;

  /**
   * @block
   * @brief Outer loop to simulate the passage of time over `t` seconds.
   */
  while (t--) {
    /**
     * @block
     * @brief Inner loop to scan the queue and perform swaps for one second.
     */
    for (int i = 0; i < n - 1; i++) {
      if (x[i] == 'B' && x[i + 1] == 'G') {
        swap(x[i], x[i + 1]);
        /**
         * @inline
         * @brief Skip the next position after a swap.
         * @details This is the critical step to ensure correctness. By
         * incrementing `i` again, we move past the two just-swapped
         * characters, preventing a single child from moving more than one
         * position per time step.
         */
        i++;
      }
    }
  }

  // Print the final arrangement of the queue.
  cout << x << endl;
  return 0;
}
