/**
 * @file 266B.CPP
 * @brief Discrete-time simulation of a state-reordering process in a queue based on local adjacency rules.
 *
 * Algorithm: Iterative single-pass swap simulation. In each time step, adjacent 'B' (boy) and 'G' (girl)
 *            pairs are swapped if 'B' precedes 'G'.
 * Time Complexity: O(T * N) where T is the number of time steps and N is the queue length.
 * Space Complexity: O(N) to store the queue elements.
 */
#include <iostream> // Required for input/output operations (std::cin, std::cout).

/**
 * @brief Simulates the swapping process in the queue over a specified number of time steps.
 * @param values A pointer to the character array representing the queue.
 * @param n The length of the queue.
 * @param t The number of time steps for the simulation.
 * @pre 'values' is a valid character array representing the queue, 'n' is its length, and 't' is the number of simulation steps.
 * @post The 'values' array is modified in place to reflect the queue's state after 't' time steps.
 */
void changePositions(char *values, int n, int t) {
  // Block Logic: Outer loop for simulating 't' time steps.
  while (true) {
    // Pre-condition: Checks if there are remaining time steps.
    // Invariant: If 't' reaches 0, the simulation is complete.
    if (t == 0)
      return; // All time steps completed, return.

    /**
     * Block Logic: Performs a single-pass sweep to resolve priority inversions ('B' followed by 'G').
     * Invariant: In each iteration, 'i' scans the queue. If a "BG" pair is found, they are swapped,
     *            and 'i' is incremented to skip the newly swapped 'B' (which just moved) to prevent
     *            it from moving multiple positions in a single time step.
     */
    for (int i = 0; i < n - 1; ++i) { // Iterates up to n-2 to access values[i+1].
      // Pre-condition: Checks for an adjacent 'B' and 'G' pair.
      if ((values[i] == 'B') && (values[i + 1] == 'G')) {
        values[i] = 'G';     // Swap 'B' with 'G'.
        values[i + 1] = 'B'; // Complete the swap.
        ++i;                 // Inline: Skips the next index to prevent the 'B' from migrating more than one position per time step.
      }
    }
    --t; // Decrement the remaining time steps.
  }
}

/**
 * @brief Main function to handle input, initiate queue simulation, and print the result.
 * @pre Input consists of an integer 'n' (queue length), 't' (time steps), followed by 'n' characters
 *      ('B' or 'G') representing the initial queue state.
 * @post Prints the final state of the queue after 't' simulation steps.
 */
int main() {
  int n, t; // 'n': length of the queue, 't': number of time steps.
  // Functional Utility: Reads the queue length and the number of time steps.
  std::cin >> n >> t;

  // Functional Utility: Dynamically allocates memory for the character array representing the queue.
  char *values = new char[n];

  // Functional Utility: Reads the initial state of the queue character by character.
  for (int i = 0; i < n; ++i) {
    std::cin >> values[i];
  }

  // Functional Utility: Calls the simulation function to reorder the queue.
  changePositions(values, n, t);

  // Functional Utility: Prints the final state of the queue.
  for (int i = 0; i < n; ++i) {
    std::cout << values[i];
  }
  // Prints a newline character at the end of the output.
  std::cout << '\n';

  // Functional Utility: Deallocates the dynamically allocated memory to prevent memory leaks.
  delete[] values;

  // Indicates successful program execution.
  return 0;
}
