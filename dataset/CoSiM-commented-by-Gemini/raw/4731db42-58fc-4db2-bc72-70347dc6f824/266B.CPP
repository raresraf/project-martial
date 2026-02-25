/**
 * @file 266B.CPP
 * @brief Discrete-time simulation of a queue reordering process.
 * @details This program simulates the evolution of a queue of boys ('B') and girls ('G')
 * over 't' seconds. In each second, any adjacent pair of a boy followed by a girl ('BG')
 * will swap their positions.
 *
 * Algorithm:
 * This is an iterative single-pass swap simulation.
 * 1. The outer loop simulates 't' discrete time steps.
 * 2. The inner loop sweeps through the queue for each time step.
 * 3. During the sweep, it identifies adjacent 'BG' pairs.
 * 4. When a 'BG' pair is found, they are swapped to 'GB'.
 * 5. A crucial step is to advance the loop counter by an extra increment (`++i`) after a swap.
 *    This ensures that an individual only moves one position per time step, maintaining
 *    the atomicity of movement within a single second of the simulation.
 *
 * Time Complexity: O(T * N), where T is the number of time steps and N is the length of the queue.
 * Space Complexity: O(N) to store the state of the queue.
 */
// input
// 5 1
// BGGBG
// output
// GBGGB

#include <bits/stdc++.h>
using namespace std;

int main() {

  /**
   * @note This preprocessor block is a common pattern in competitive programming
   * for redirecting standard I/O to files for local testing. It has no effect
   * when the code is compiled with the ONLINE_JUDGE flag, as is typical on
   * judging servers.
   */
#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif

  int n, t;
  cin >> n >> t;
  string a;
  cin >> a;

  /**
   * @block
   * @description This loop orchestrates the temporal progression of the simulation,
   * running for 't' seconds.
   * Invariant: At the start of each iteration, string 'a' represents the queue's
   * state at the beginning of the current time step.
   */
  while (t-- > 0) {
    /**
     * @block
     * @description This loop performs a single-pass sweep across the queue to
     * resolve all possible swaps for the current time step.
     */
    for (int i = 1; i < n; i++) {
      // Check for a boy followed by a girl.
      if (a[i] == 'G' && a[i - 1] == 'B') {
        // Perform the swap.
        a[i] = 'B';
        a[i - 1] = 'G';
        /**
         * @inline
         * @description Skips the next position in the queue. This is essential to
         * ensure that a child who has just moved does not move again in the same
         * time step, enforcing atomic movement.
         */
        ++i;
      }
    }
  }
  // Output the final state of the queue after 't' seconds.
  cout << a;
  return 0;
}