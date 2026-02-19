/**
 * @file 266B.CPP
 * @brief Solution to the "Queue at the School" problem.
 * @details This program simulates a queue of boys ('B') and girls ('G'). In each time step,
 * any boy standing directly in front of a girl swaps places with her. The program
 * calculates the state of the queue after a given amount of time.
 *
 * Author: Ikki
 * Date: Jun 11, 2015
 *
 * Algorithm: Discrete Time Simulation
 * 1. The state of the queue at the beginning of each time step is stored in string `a`.
 * 2. A temporary string `b` is used to build the state of the queue for the *next* time step.
 * 3. The simulation runs for `t` time steps (the outer loop).
 * 4. In each time step, the program iterates through the current queue `a`.
 * 5. If it finds a 'B' followed by a 'G' (`a[i] == 'B' && a[i+1] == 'G'`), it swaps them in the
 *    temporary string `b`. The loop index `i` is then incremented by 2 to move past this pair.
 *    This is a critical step to ensure that a single swap happens for this pair in this time step
 *    and prevents the newly moved girl from causing another immediate swap.
 * 6. If no 'BG' pair is found, the character is copied as is from `a` to `b`, and `i` is incremented by 1.
 * 7. After the inner loop completes, `b` holds the new state of the queue. This state is copied
 *    back to `a` (`a = b;`) to prepare for the next time step.
 * 8. After `t` steps, the final queue configuration is printed.
 *
 * Time Complexity: O(t * n), where `t` is the number of time steps and `n` is the length of the queue.
 * Space Complexity: O(n), to store the two strings representing the queue states.
 */
#include <iostream>
#include <string>
using namespace std;

int main() {
  int n, t;
  cin >> n >> t;
  string a, b;
  cin >> a;
  b = a; // 'a' holds the state at the start of a time step, 'b' is used to build the next state.

  /**
   * Block Logic: Simulate the queue for 't' seconds.
   * Pre-condition: `a` contains the initial queue state.
   * Invariant: At the end of each outer loop iteration, `a` is updated to the new state of the queue after one second.
   */
  for (int tt = 0; tt < t; tt++) {
    /**
     * Block Logic: A single pass to perform all swaps for the current time step.
     * Changes are based on string 'a' and applied to string 'b'.
     */
    for (int i = 0; i < n - 1;) {
      if (a[i] == 'B' && a[i + 1] == 'G') {
        // Swap the 'B' and 'G' in the temporary string 'b'.
        b[i] = 'G';
        b[i + 1] = 'B';
        // Skip past the swapped pair.
        i += 2;
      } else {
        // If no swap, just copy the character from the current state.
        b[i] = a[i];
        i++;
      }
    }
    // Update the current state 'a' with the new state from 'b' for the next time step.
    a = b;
  }
  
  // Print the final state of the queue.
  cout << b << endl;
  return 0;
}
