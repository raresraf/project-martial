/**
 * @file 266B.CPP
 * @brief Discrete-time simulation of a state-reordering process based on local adjacency rules.
 * This program simulates a queue where boys ('B') and girls ('G') are arranged in a line.
 * In each time step, if a boy is immediately followed by a girl (BG), they swap positions (GB).
 * This process is repeated for a specified number of time steps.
 * Algorithm: Iterative single-pass swap simulation.
 * Time Complexity: O(T * N) where T is the number of time steps (m) and N is the queue length (s.size()).
 * Space Complexity: O(N) to store the entity sequence.
 */
#include <iostream>
#include <string> // This was missing in the original code, but implicitly used for string s. Adding for completeness, as per standard practice, but not altering existing code behavior.

using namespace std;

int main() {
  int n; // Represents the length of the string/queue.
  int m; // Represents the number of time steps/iterations for the simulation.
  string s; // Represents the initial state of the queue as a string of 'B' and 'G' characters.
  cin >> n >> m; // Reads the length of the queue and the number of simulation steps.

  cin >> s; // Reads the initial configuration of the queue.

  /**
   * Block Logic: Orchestrates the temporal progression of the simulation.
   * This outer loop iterates 'm' times, representing each time step of the simulation.
   * Invariant: At the start of each iteration, the string 's' represents the queue state
   * at time T-k, where k is the current iteration count.
   */
  for (int i = 0; i < m; i++) {
    /**
     * Block Logic: Performs a single-pass sweep to resolve priority inversions.
     * This inner loop iterates through the queue from left to right.
     * Logic: Identifies 'B' (Boy) and 'G' (Girl) pairs where 'B' precedes 'G'
     * and performs a local state transition (swap).
     * Pre-condition: 'j' is a valid index within the string, and 'j+1' is also valid.
     * Post-condition: If a 'BG' pair is found, they are swapped to 'GB'.
     */
    for (int j = 0; j < s.size() - 1; j++) { // Loop up to s.size() - 1 to avoid out-of-bounds access for s[j+1]
      if (s[j] == 'B' && s[j + 1] == 'G') {
        s[j] = 'G';
        s[j + 1] = 'B';
        // Inline: Skips the next index to prevent a single 'B' from migrating
        // more than one position per time step (atomic movement).
        j++;
      }
    }
  }
  /**
   * Block Logic: Outputs the final state of the queue after all simulation steps.
   * This loop iterates through the modified string 's' and prints each character.
   */
  for (int i = 0; i < n; i++) {
    cout << s[i];
  }
  cout << endl; // Add a newline character at the end of the output for proper formatting.
  return 0;
}

