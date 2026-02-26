/**
 * @file 266B.CPP
 * @brief Solves the "Queue at the School" problem from Codeforces.
 * @author Nikhil Nagrale (nikhilnagrale2)
 *
 * @details This program simulates the changing arrangement of a queue of boys
 * ('B') and girls ('G'). Over `t` seconds, any boy standing immediately
 * before a girl will swap places with her. The program determines the final
 * state of the queue.
 *
 * @algorithm
 * A direct, discrete-time simulation is performed.
 * 1. The simulation runs for `t` time steps (seconds), controlled by an outer loop.
 * 2. In each step, an inner loop scans the queue from left to right.
 * 3. If a 'B' is found at position `i` and a 'G' at `i+1`, they are swapped.
 * 4. After a swap, the loop counter `i` is incremented an extra time. This is
 *    a crucial detail to ensure that the newly moved girl doesn't cause another
 *    swap in the same time step, correctly modeling that all swaps within one
 *    second happen "simultaneously".
 *
 * @complexity
 * Time Complexity: O(T * N), where T is the number of seconds to simulate and
 * N is the length of the queue.
 * Space Complexity: O(N) to store the queue arrangement in a string.
 */
// code by Nikhil Nagrale
// nikhilnagrale2 on EveryPlatform
#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, t; // n: number of children, t: time in seconds
  cin >> n >> t;
  string a; // Represents the queue
  cin >> a;

  /**
   * @block
   * @brief Outer loop to simulate the passage of time over `t` seconds.
   */
  while (t--) {
    /**
     * @block
     * @brief Inner loop to scan the queue and perform swaps for one second.
     */
    for (int i = 0; i < n - 1; i++)
      if (a[i] == 'B' && a[i + 1] == 'G') {
        swap(a[i], a[i + 1]);
        /**
         * @inline
         * @brief Skip the next position after a swap.
         * @details This prevents the just-swapped 'G' at position `i` from
         * being considered again in this same time step. It also prevents the
         * 'B' at `i+1` from causing a double-move in one second. This correctly
         * models the simultaneous nature of the swaps.
         */
        i++;
      }
  }

  // Print the final arrangement of the queue.
  cout << a << endl;
  return 0;
}
