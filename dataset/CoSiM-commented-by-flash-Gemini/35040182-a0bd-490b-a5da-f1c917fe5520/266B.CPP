/**
 * @file 266B.CPP
 * @brief Solution for a competitive programming problem simulating a queue rearrangement process.
 *
 * Algorithm: Iterative simulation of a queue over `t` time steps. In each time step,
 *            a linear scan of the queue is performed. If a 'B' (boy) is immediately
 *            followed by a 'G' (girl), they swap positions. To ensure that each
 *            'B' only moves once per time step, the inner loop's index `j` is incremented
 *            an additional time after a swap.
 * Time Complexity: O(T * N) where T is the number of time steps and N is the length of the queue.
 * Space Complexity: O(N) for storing the queue string.
 */

#include <iostream>
#include <string> // Include <string> for std::string and std::swap
#include <algorithm> // Include <algorithm> for std::swap

/**
 * @brief Main entry point of the program.
 *
 * Functional Utility: Simulates the rearrangement of a queue of boys ('B')
 *                     and girls ('G') over `t` time steps. In each step,
 *                     boys move past girls if they are immediately ahead.
 *                     The final state of the queue is printed to standard output.
 *
 * @return 0 on successful execution.
 */
int main() {
  int n; // @brief The number of elements in the queue (length of the string).
  int t; // @brief The number of time steps (iterations) for the simulation.
  std::string queue; // @brief The string representing the queue of 'B's and 'G's.
  std::cin >> n >> t;
  std::cin >> queue;

  // Block Logic: Simulate the queue rearrangement over `t` time steps.
  // Invariant: After each iteration, one full pass of potential swaps has occurred.
  for (int i = 0; i < t; i++) {
    // Block Logic: Iterate through the queue to find and perform swaps.
    // Invariant: `j` traverses the queue. Swaps are performed in-place.
    //            `j` is incremented an extra time after a swap to ensure
    //            a boy only moves one position per time step.
    for (int j = 0; j < n; j++) {
      // Block Logic: Check if a 'B' is followed by a 'G'.
      // Precondition: `j+1` is a valid index within the queue.
      // Invariant: If `queue[j]` is 'B' and `queue[j+1]` is 'G', they are swapped.
      if (queue[j] == 'B' && queue[j + 1] == 'G') {
        std::swap(queue[j], queue[j + 1]); // Functional Utility: Swap the 'B' and 'G' characters.
        j++; // Functional Utility: Skip the next character to prevent double-swapping 'B' in the same pass.
      }
    }
  }

  // Functional Utility: Print the final state of the queue after all time steps.
  std::cout << queue;

  return 0;
}