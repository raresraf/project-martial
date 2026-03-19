/**
 * @file raw/c6f2bf9d-d11c-4230-91d4-ab68648d9776/266B.CPP
 * @brief Simulates a queue reordering process over discrete time steps.
 *
 * This program models a queue of 'B's (Boys) and 'G's (Girls). Over `t` seconds,
 * any 'B' standing immediately in front of a 'G' swaps places with her. This
 * implementation performs all swaps for a given time step based on the queue's
 * state at the beginning of that step (a parallel update).
 *
 * @algorithm
 * 1. An outer loop iterates `t` times, representing `t` seconds.
 * 2. An inner loop scans through the current state of the queue.
 * 3. A separate `result` string is used to store the outcome of the swaps.
 * 4. If a 'B' is found before a 'G' in the original `queue` string, their
 *    positions are swapped in the `result` string.
 * 5. After the inner loop completes, the original `queue` is updated with the
 *    content of `result`, preparing it for the next time step.
 *
 * @time_complexity O(T * N), where T is the number of time steps and N is the
 *                  length of the queue.
 * @space_complexity O(N) to store a copy of the queue string for swaps.
 */
#include <iostream>
#include <string>
#include <vector>

using namespace std;

int main(void) {
  int n, t;
  string queue, result;
  cin >> n >> t;
  cin >> queue;
  result = queue; // `result` will store the state after one time step's swaps.

  /**
   * @block
   * Outer loop for temporal progression, simulating `t` seconds.
   */
  for (int i = 0; i < t; i++) {
    /**
     * @block
     * Inner loop to scan the queue and identify all possible swaps for the
     * current time step.
     * @warning The loop variable `i` here shadows the outer loop's `i`.
     *          This is confusing and poor practice, but functionally correct.
     */
    for (int i = 0; i < queue.size() - 1; i++) {
      /**
       * @logic
       * This check is always performed on the original `queue` state from the
       * beginning of the time step. The swaps are written to `result`, leading
       * to a parallel-update behavior where all swaps for a time step happen
       * "simultaneously", rather than sequentially.
       */
      if (queue.at(i) == 'B' && queue.at(i + 1) == 'G') {
        result.at(i) = queue.at(i + 1);
        result.at(i + 1) = queue.at(i);
      }
    }

    // Update the queue state for the next time step.
    queue = result;
  }

  cout << result << endl;
  return 0;
}
