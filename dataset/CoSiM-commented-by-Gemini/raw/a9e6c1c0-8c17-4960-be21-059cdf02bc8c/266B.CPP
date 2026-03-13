/**
 * @file 266B.CPP
 * @brief Solution for CodeForces Problem 266B - "Queue at the School".
 *
 * @details
 * This program simulates a queue of children, represented by 'B' for boys and 'G' for girls.
 * In each time step, any boy ('B') standing directly in front of a girl ('G') swaps places with her.
 * The simulation runs for a specified number of time steps (`seconds`).
 *
 * @algorithm Iterative in-place single-pass swap
 * @time_complexity O(T * N), where T is the number of time steps (`seconds`) and N is the
 *                  length of the queue (`kids`).
 * @space_complexity O(N) to store the queue.
 *
 * @note The use of a C-style Variable Length Array (`char order[kids]`) is a non-standard
 *       C++ extension. A `std::string` or `std::vector<char>` would be standard C++.
 */
#include <cstdio>
#include <iostream>

using namespace std;

int main() {
  int kids, seconds;
  cin >> kids >> seconds;

  char order[kids + 1]; // Use kids+1 for null terminator safety with `cin >> order`.
  cin >> order;

  /**
   * @brief This outer loop orchestrates the temporal progression of the simulation,
   * running for the specified number of `seconds`.
   */
  for (int i = 0; i < seconds; i++) {
    /**
     * @brief This inner loop performs a single-pass sweep to resolve 'BG' inversions.
     * It iterates through the queue to find adjacent 'B' and 'G' pairs and swaps them.
     */
    for (int j = 0; j < kids - 1; j++) {
      // Pre-condition: Check if a boy is immediately followed by a girl.
      if (order[j] == 'B' && order[j + 1] == 'G') {
        // Perform the local state transition (swap).
        order[j + 1] = 'B';
        order[j] = 'G';
        /**
         * @brief Inline Logic: Skips the next index.
         * This is crucial to ensure that a single boy does not move more than one position
         * to the right in a single time step, preserving the atomic nature of the movement.
         */
        j++;
      }
    }
  }

  // Output the final state of the queue after the simulation.
  cout << order;

  return 0;
}
