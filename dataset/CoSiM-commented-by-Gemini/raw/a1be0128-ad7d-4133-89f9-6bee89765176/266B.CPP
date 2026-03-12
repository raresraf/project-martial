/**
 * @file 266B.CPP
 * @brief Solution to the "Queue at the School" problem (Codeforces 266B).
 * @details This program simulates the changes in a queue of boys ('B') and girls ('G')
 *          over a given period of time. In each second, any boy standing immediately
 *          in front of a girl swaps places with her.
 *
 * @algorithm The simulation is performed iteratively.
 *            1. An outer loop runs for 't' seconds (the total time).
 *            2. An inner loop scans the queue for adjacent 'B' and 'G' pairs ('BG').
 *            3. When a 'BG' pair is found, they are swapped to 'GB'.
 *            4. To correctly model the simultaneous nature of swaps in a single time
 *               step with a single pass, the index is advanced by 2 after a swap.
 *               This prevents the newly moved 'B' from swapping again in the same time step.
 *            5. The process is repeated for 't' seconds to get the final queue arrangement.
 *
 * @time_complexity O(N * T), where N is the length of the queue and T is the time in seconds.
 * @space_complexity O(N) to store the queue string.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {

  int n, t;

  string s;

  scanf("%d%d", &n, &t);

  cin >> s;

  /**
   * @brief Outer loop simulates the passage of time, from 1 to 't' seconds.
   */
  for (int i = 1; i <= t; ++i) {
    /**
     * @brief Inner loop scans the queue to perform swaps for the current second.
     * The loop control `j` is manually incremented inside the loop.
     */
    for (int j = 0; j < n - 1;) {

      // Check for a boy 'B' standing immediately before a girl 'G'.
      if (s[j] == 'B' && s[j + 1] == 'G') {
        swap(s[j], s[j + 1]);
        /**
         * @brief Advance index by 2 after a swap. This is crucial because the boy
         * at the new position (j+1) has already had his turn to swap in this
         * time step. This correctly models the simultaneous nature of the swaps
         * within a single linear scan.
         */
        j += 2;
      } else {
        // If no swap occurs, move to the next position.
        j++;
      }

      // cout << "       " << i << "    " << j << endl;
    }

    // cout << s << endl;
  }

  // Print the final state of the queue.
  cout << s << endl;
}