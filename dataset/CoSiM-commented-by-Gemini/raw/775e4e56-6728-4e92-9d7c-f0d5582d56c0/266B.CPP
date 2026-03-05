/**
 * @file 266B.CPP
 * @brief Solves the "Queue at the School" problem.
 *
 * This program simulates the arrangement of a queue of boys ('B') and girls ('G')
 * over a period of `t` seconds. In each second, every boy standing immediately
 * in front of a girl swaps places with her.
 *
 * @algorithm
 * The simulation proceeds for `t` time steps. In each time step, a pass is made
 * through the queue to perform all possible swaps for that second.
 *
 * 1. Read `n` (the number of children), `t` (the time in seconds), and the
 *    initial queue arrangement string `s`.
 * 2. Loop `t` times to simulate the passage of time.
 * 3. Inside the time loop, perform a single pass of swaps for one second.
 *    - The pass iterates from right to left (from index `n-1` down to 1).
 *    - At each position `i`, it checks if `s[i]` is a girl ('G') and `s[i-1]` is
 *      a boy ('B').
 *    - If this 'BG' pattern is found, the characters are swapped.
 *    - After a swap, the loop counter `i` is manually decremented. This is a
 *      key part of the logic: it ensures that after swapping `s[i-1]` and `s[i]`,
 *      the next character to be checked is at index `i-2`. This prevents the
 *      newly placed boy at `s[i]` from being immediately considered for another
 *      swap in the same time step, correctly modeling that each swap takes one second.
 * 4. After `t` seconds, print the final state of the queue string `s`.
 *
 * @complexity
 * Time Complexity: O(t * n), where `t` is the number of seconds and `n` is the
 * length of the queue.
 * Space Complexity: O(n) to store the queue string.
 */

#include <iostream>
#include <string>
// The <memory.h> header is not necessary for this code.
#include <memory.h>

using namespace std;

int main() {
  // n: number of children in the queue
  // t: time in seconds
  int n, t;
  string s;
  cin >> n >> t >> s;

  /**
   * @brief Outer loop for simulating `t` seconds.
   */
  while (t--)
    /**
     * @brief Inner loop for performing one second's worth of swaps.
     * Iterating from right to left simplifies the logic, as we don't have to
     * worry about a boy moving multiple times in a single pass.
     */
    for (int i = n - 1; i > 0; i--)
      // Check for a 'B' followed by a 'G'.
      if (s[i] == 'G' && s[i - 1] == 'B') {
        // Swap the boy and girl.
        s[i] = 'B';
        s[i - 1] = 'G';
        // Decrement `i` to skip the position of the just-swapped boy.
        // The boy at the original `i-1` position is now at `i` and should not
        // be considered for another swap in this same time step.
        i--;
      }

  cout << s << endl;

  return 0;
}
