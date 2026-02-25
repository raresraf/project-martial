/**
 * @file 266B.CPP
 * @brief Solution to Codeforces problem 266B (Queue at the School).
 *
 * @details
 * This program simulates the changing arrangement of a queue of boys ('B') and
 * girls ('G') over a period of time. The simulation follows a simple rule: in
 * each second, if a boy is standing immediately in front of a girl, they swap
 * places.
 *
 * @section algorithm
 * The problem is a discrete-time simulation of a state-reordering process based
 * on a local swap rule.
 *
 * The algorithm is implemented as follows:
 * 1. An outer loop runs for `t` iterations, where `t` is the total time in
 *    seconds for the simulation.
 * 2. An inner loop iterates through the queue for each second.
 * 3. Inside the inner loop, it looks for adjacent pairs of 'B' and 'G' (a boy
 *    in front of a girl).
 * 4. When a "BG" pair is found at indices `i` and `i+1`, their positions are
 *    swapped.
 * 5. After a swap, the loop index `i` is incremented by 2. This is a crucial
 *    step to ensure that a child moves at most once per second, correctly
 *    simulating simultaneous swaps. For example, in the queue `BGB`, after
 *    swapping the last two to get `BBG`, the now-adjacent `BG` should not be
 *    swapped again in the same time step.
 * 6. If no swap occurs at the current position, the index `i` is incremented by 1.
 * 7. After the outer loop completes, the final arrangement of the queue is printed.
 *
 * @section complexity
 * - Time Complexity: O(T * N), where T is the number of seconds and N is the
 *   length of the queue.
 * - Space Complexity: O(N) to store the queue arrangement in a string.
 */

#include <iostream>
#include <string>
using namespace std;

int main() {
  // n: the number of children in the queue.
  // t: the time in seconds to run the simulation.
  int n, t;
  string A; // The string representing the queue.
  cin >> n >> t;
  cin >> A;

  /**
   * @brief Outer loop to simulate the passage of time.
   *
   * Invariant: At the beginning of each iteration, `A` represents the queue's
   * state at the current time step.
   */
  while (t--) {
    int i = 0;
    /**
     * @brief Inner loop to perform swaps for a single time step.
     *
     * This loop sweeps through the queue and performs all possible "BG" swaps.
     */
    while (i < (n - 1)) {
      // Pre-condition: Check for a boy followed by a girl.
      if (A[i] == 'B' && A[i + 1] == 'G') {
        // Perform the swap.
        A[i] = 'G';
        A[i + 1] = 'B';
        // Advance index by 2 to prevent the same children from moving again
        // in the same time step. This correctly models simultaneous swaps.
        i += 2;
      } else {
        // If no swap, move to the next position.
        i++;
      }
    }
  }

  // Print the final state of the queue after t seconds.
  cout << A;
  return 0;
}
