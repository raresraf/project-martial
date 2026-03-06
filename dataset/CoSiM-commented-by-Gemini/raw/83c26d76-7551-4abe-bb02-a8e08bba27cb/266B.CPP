/**
 * @file 266B.CPP
 * @brief Simulates a queue of boys and girls swapping positions over time.
 * @details This program solves the "Queue at the School" problem from Codeforces.
 * It simulates a process where in each time step, any boy ('B') standing
 * immediately in front of a girl ('G') swaps places with her. This is repeated
 * for a specified number of time steps.
 *
 * @algorithm Iterative single-pass swap simulation.
 * @complexity
 * Time: O(T * N), where T is the number of time steps and N is the length of the queue.
 *       The nested loops dictate the complexity.
 * Space: O(N) to store the state of the queue as a string.
 */
#include <iostream>
#include <string>
#include <vector> // Included for completeness, though not directly used.

using namespace std;

/**
 * @brief Main function to handle input and run the queue simulation.
 */
int main() {
  string s, a; // s: initial queue state, a: active queue state for simulation.
  int n, t;    // n: number of children, t: number of time steps.
  int i, j;
  cin >> n >> t;
  cin >> s;
  a = s;

  /**
   * Block Logic: Orchestrates the temporal progression of the simulation.
   * Invariant: At the start of each outer loop iteration, string 'a' represents
   * the queue's state at the beginning of that time step.
   */
  for (i = 0; i < t; i++) {
    /**
     * Block Logic: Performs a single-pass sweep through the queue to resolve
     * priority inversions ('B' before 'G').
     */
    for (j = 1; j < n; j++) {
      // Find adjacent 'B' and 'G' pairs.
      if (a[j - 1] == 'B' && a[j] == 'G') {
        // Perform the state transition (swap).
        a[j - 1] = 'G';
        a[j] = 'B';
        // Inline: Skips the next index to ensure a boy moves at most one
        // position forward in a single time step (atomic movement).
        j++;
      }
    }
  }
  
  // Output the final state of the queue after 't' time steps.
  cout << a;
  
  return 0;
}
