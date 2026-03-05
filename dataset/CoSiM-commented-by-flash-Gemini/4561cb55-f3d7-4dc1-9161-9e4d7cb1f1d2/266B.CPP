/**
 * @file 266B.CPP
 * @brief Discrete-time simulation of a state-reordering process based on local adjacency rules.
 *
 * This program simulates the reordering of 'B' (boy) and 'G' (girl) characters
 * in a queue. In each time step, if a 'B' is followed by a 'G', they swap positions.
 * The simulation runs for a specified number of time steps.
 *
 * Algorithm: Iterative single-pass swap simulation.
 * Time Complexity: O(T * N) where T is the number of time steps and N is the queue length.
 * Space Complexity: O(N) to store the character array.
 */
#include <stdio.h>

/**
 * @brief Main function to execute the queue reordering simulation.
 *
 * Reads the queue length (n), number of time steps (t), and the initial queue
 * configuration from standard input. It then simulates the reordering process
 * and prints the final state of the queue.
 *
 * @return 0 on successful execution.
 */
int main() {
  // Declare short integers for queue length 'n' and number of time steps 't'.
  short n, t;
  // Read 'n' and 't' from standard input.
  scanf("%hd %hd", &n, &t);
  // Declare a character array 'arr' of size 'n + 1' to store the queue configuration.
  // The +1 is for the null terminator.
  char arr[n + 1];
  // Read the initial queue configuration string into 'arr'.
  scanf("%s", arr);

  /**
   * @brief Block Logic: Orchestrates the temporal progression of the simulation.
   *
   * This loop iterates 't' times, with each iteration representing one unit of time.
   * Invariant: At the start of each iteration, the array 'arr' represents the queue
   * state at time T-k, where k is the number of completed iterations.
   */
  while (t--) {
    /**
     * @brief Block Logic: Performs a single-pass sweep to resolve priority inversions.
     *
     * This loop iterates through the queue to find adjacent 'B' and 'G' characters
     * and swaps them.
     * Invariant: At the start of each inner iteration, characters up to 'i-1' have
     * been considered for swapping in the current time step.
     */
    for (int i = 0; i < n - 1; i++) {
      // Pre-condition: Check if the current character is 'B' and the next is 'G'.
      // This identifies a 'B' followed by a 'G', indicating a swap is needed.
      if (arr[i] == 'B' && arr[i + 1] == 'G') {
        // Perform the swap: 'B' moves to the right, 'G' moves to the left.
        char aux = arr[i];
        arr[i] = arr[i + 1];
        arr[i + 1] = aux;
        // Inline: Increment 'i' to skip the character that was just moved to the
        // current position. This ensures that a 'B' does not move more than one
        // position per time step (atomic movement) and prevents double-processing.
        i++;
      }
    }
  }
  // Print the final state of the queue after all time steps have been simulated.
  printf("%s", arr);

  return 0;
}
