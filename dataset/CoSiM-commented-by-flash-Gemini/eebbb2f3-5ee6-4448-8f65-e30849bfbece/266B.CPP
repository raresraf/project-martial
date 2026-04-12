/**
 * @file 266B.CPP
 * @brief Solution to the "Queue at the School" problem.
 *
 * This program simulates the reordering of a queue consisting of boys ('B') and girls ('G').
 * In each second, every instance of a 'B' immediately followed by a 'G' (BG) simultaneously
 * swaps to become 'GB'. The simulation runs for 't' seconds.
 *
 * Algorithm: Discrete-time iterative swap simulation.
 * Time Complexity: O(T * N) where T is the number of time steps and N is the length of the queue.
 * Space Complexity: O(N) to store the queue string.
 */
#include <iostream>
using namespace std;

/**
 * @brief Main function to simulate the queue reordering.
 *
 * Functional Utility: Reads initial queue configuration and simulation time,
 * then applies swap rules iteratively, and finally outputs the rearranged queue.
 */
int main(int argc, char *argv[]) {
  int n, t;
  string queue;
  cin >> n >> t;     // Read queue length 'n' and simulation time 't'.
  cin >> queue;      // Read the initial queue string.

  /**
   * Block Logic: Handles the edge case where the queue has only one person.
   * If N=1, no swaps can occur, so the queue remains unchanged.
   * Post-condition: Prints the queue and exits if N=1.
   */
  if (n == 1) {
    cout << queue << endl;
    return 0;
  }
  /**
   * Block Logic: Orchestrates the temporal progression of the simulation.
   * Invariant: At the start of each iteration, 'queue' represents the state after 'T-k' seconds.
   */
  while (t--) {
    /**
     * Block Logic: Performs a single-pass sweep to resolve 'BG' to 'GB' swaps.
     * Logic: Iterates through the queue, identifies 'BG' pairs, and performs a local swap.
     * The 'i++' in the conditional branch ensures that a boy who just moved
     * past a girl does not participate in another swap in the same second.
     * Invariant: After each full inner loop, all possible 'BG' swaps for the current second have been processed.
     */
    for (int i = 0; i < queue.length() - 1; i++) {
      if (queue.at(i) == 'B' && queue.at(i + 1) == 'G') {
        queue[i] = 'G';
        queue[i + 1] = 'B';
        i++; // Skip the next character as it was just swapped.
      }
    }
  }
  cout << queue << endl; // Output the final state of the queue.
  return 0;
}
