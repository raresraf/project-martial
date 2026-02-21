/**
 * @file 266B.CPP
 * @brief Simulates the rearrangement of a queue of boys and girls over time.
 *
 * This program models a queue composed of boys ('B') and girls ('G'). Over a
 * series of discrete time steps, any boy standing immediately in front of a
 * girl swaps places with her. This process is repeated for a specified number
 * of time steps.
 *
 * Algorithm:
 * 1. Read the number of children `n` and the number of time steps `m`.
 * 2. Read the initial arrangement of the queue as a string `s`.
 * 3. An outer loop simulates the passage of time, running `m` times.
 * 4. An inner loop performs a single pass over the queue for each time step.
 * 5. Inside the inner loop, if a "BG" pair is found (a boy in front of a girl),
 *    their positions are swapped.
 * 6. After a swap, the inner loop counter `j` is incremented. This is a
 *    critical step to ensure that a single boy moves at most one position to
 *    the right per time step, correctly modeling the discrete nature of the
 *    simulation.
 * 7. After the outer loop completes, the final state of the queue is printed.
 *
 * Time Complexity: O(M * N), where M is the number of time steps and N is the
 * number of children in the queue.
 * Space Complexity: O(N) to store the queue string.
 */
#include <iostream>
#include <string>

using namespace std;

int main() {
  int n; // Number of children
  int m; // Number of time steps
  string s;
  cin >> n >> m;
  cin >> s;
  
  /**
   * @brief Outer loop for simulating `m` time steps.
   * Pre-condition: `s` holds the queue state from the previous time step.
   */
  for (int i = 0; i < m; i++) {
    /**
     * @brief Inner loop to perform one pass of swaps for a single time step.
     */
    for (int j = 0; j < s.size(); j++) {
      // Check for a Boy-Girl pair that needs to be swapped.
      if (s[j] == 'B' && s[j + 1] == 'G') {
        s[j] = 'G';
        s[j + 1] = 'B';
        // Increment `j` to skip the next position. This prevents the
        // newly-placed 'G' from being considered for another swap in the
        // same pass and ensures each 'B' moves only one spot right.
        j++;
      }
    }
  }
  
  // Print the final arrangement of the queue.
  for (int i = 0; i < n; i++) {
    cout << s[i];
  }
  return 0;
}
