/**
 * @file 266B.CPP
 * @brief Solves the "Queue at the School" problem from Codeforces.
 *
 * @details This program simulates the arrangement of a queue of boys ('B') and
 * girls ('G'). For a given time `t`, the simulation swaps any boy standing
 * immediately in front of a girl. This process is repeated `t` times.
 *
 * @algorithm
 * A direct, discrete-time simulation is performed.
 * 1. The main simulation loop runs for `t` iterations, each representing one
 *    second.
 * 2. In each iteration, a nested loop scans the queue from left to right.
 * 3. If a 'B' is found at position `j` and a 'G' at `j+1`, the two characters
 *    are swapped.
 * 4. To correctly model that all swaps in a single second happen
 *    "simultaneously", the inner loop index `j` is incremented an extra
 *    time after a swap. This prevents the same characters from being involved
 *    in multiple swaps within the same time step.
 *
 * @complexity
 * Time Complexity: O(T * N), where T is the time in seconds and N is the
 * length of the queue.
 * Space Complexity: O(N) to store the queue as a string.
 */
#include <bits/stdc++.h>
#include <string>

using namespace std;

int main() {
  int i, j, n, t;
  string s;

  // Read number of children, time, and the initial queue arrangement.
  cin >> n >> t >> s;

  /**
   * @block
   * @brief Outer loop for simulating `t` seconds of time.
   */
  for (i = 1; i <= t; i++) {
    /**
     * @block
     * @brief Inner loop to scan the queue and perform swaps for one second.
     */
    for (j = 0; j < n; j++) {
      // Check for a Boy ('B') followed by a Girl ('G').
      if (s[j] == 'B' && s[j + 1] == 'G') {
        swap(s[j], s[j + 1]);
        /**
         * @inline
         * @brief Advance index post-swap.
         * This is the critical step to ensure correctness. By incrementing `j`
         * again, we move past the two just-swapped characters, preventing a
         * single child from moving more than one position per time step.
         */
        j++;
      }
    }
  }

  // Print the final state of the queue.
  cout << s << endl;

  return 0;
}

// 666Lucifer666