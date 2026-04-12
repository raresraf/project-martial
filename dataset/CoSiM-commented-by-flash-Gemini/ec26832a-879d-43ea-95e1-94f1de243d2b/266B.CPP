/**
 * @file 266B.CPP
 * @brief Solution to the "Queue at the School" problem.
 *
 * This program simulates the reordering of a queue consisting of boys ('B') and girls ('G').
 * In each second, every instance of a 'B' immediately followed by a 'G' (BG) simultaneously
 * swaps to become 'GB'. The simulation runs for 't' seconds.
 *
 * Algorithm: Discrete-time iterative swap simulation using a character vector.
 * Time Complexity: O(T * N) where T is the number of time steps and N is the length of the queue.
 * Space Complexity: O(N) to store the queue in a vector.
 */
#include <bits/stdc++.h>

using namespace std;

/**
 * @brief Main function to simulate the queue reordering.
 *
 * Functional Utility: Reads initial queue configuration and simulation time,
 * then applies swap rules iteratively, and finally outputs the rearranged queue.
 */
int main() {
  int n, t;       // 'n' for queue length, 't' for simulation time.
  string input;   // Temporary string to read the initial queue.

  cin >> n >> t;    // Read queue length and simulation time.
  cin >> input;     // Read the initial queue string.
  // Convert the input string into a vector of characters for easier manipulation.
  vector<char> queue(input.begin(), input.end());

  /**
   * Block Logic: Orchestrates the temporal progression of the simulation.
   * The loop runs for 't' seconds, decrementing 't' in each iteration.
   * Invariant: At the start of each iteration, 'queue' represents the state after 'T-k' seconds.
   */
  while (t > 0) {
    /**
     * Block Logic: Performs a single-pass sweep to resolve 'BG' to 'GB' swaps.
     * Logic: Iterates through the queue from left to right. If a 'B' is followed by a 'G',
     * they swap positions. The 'i++' after a swap ensures that a boy who just moved
     * past a girl does not participate in another swap in the same second,
     * maintaining the "simultaneous" aspect of the problem.
     * Pre-condition: 'queue' contains the current state of the queue.
     * Invariant: After each full inner loop, all possible 'BG' swaps for the current second have been processed.
     */
    for (int i = 1; i < queue.size(); i++) {
      if (queue[i] == 'G' && queue[i - 1] == 'B') { // Found a 'BG' pattern (B at i-1, G at i)
        swap(queue[i], queue[i - 1]);                // Perform the swap to 'GB'.
        i++; // Increment 'i' to skip the newly swapped 'B' to prevent double-swaps in one second.
      }
    }
    t--; // Decrement remaining time.
  }

  /**
   * Block Logic: Prints the final state of the queue after all simulation steps.
   * Post-condition: The final rearranged queue is printed to standard output, followed by a newline.
   */
  for (char c : queue) {
    cout << c;
  }
  cout << '\n';
  return 0;
}