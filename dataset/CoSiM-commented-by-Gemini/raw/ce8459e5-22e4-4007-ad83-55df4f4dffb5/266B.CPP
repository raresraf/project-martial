/**
 * @file 266B.CPP
 * @brief Solution for Codeforces Problem 266B, "Queue at the School".
 * @details This program simulates the arrangement of a queue of boys ('B')
 *          and girls ('G') after `t` seconds. In each second, any boy
 *          standing immediately in front of a girl swaps places with her.
 *
 * @algorithm
 * The simulation is performed iteratively. An outer loop runs for `t` seconds.
 * Inside, an inner loop traverses the queue. When an adjacent 'B' followed by
 * a 'G' is found, they are swapped. The inner loop's counter is then
 * incremented an extra time (`i++`) to ensure that a single entity does not
 * move more than one position in a single time step, which correctly models
 * the simultaneous nature of the swaps described in the problem.
 *
 * @complexity
 * Time: O(n * t), where `n` is the number of children and `t` is the time.
 * Space: O(n) to store the queue string.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {

  string s;
  char f, l; // Unused variables.
  long int i, j, x, m, t, n; // j, m are unused.
  
  // The while loop is structured to handle multiple test cases,
  // though typically only one is provided.
  while (cin >> n >> t) {
    cin >> s;

    /**
     * @brief This outer loop simulates the passage of `t` seconds.
     */
    for (x = 1; x <= t; x++) {
      /**
       * @brief This inner loop performs a single pass over the queue,
       *        representing one second of simultaneous swaps.
       */
      for (i = 1; i < s.size(); i++) {
        // A boy 'B' is in front of a girl 'G'.
        if (s[i] == 'G' && s[i - 1] == 'B') {
          // Perform the swap.
          s[i] = 'B';
          s[i - 1] = 'G';
          /**
           * @brief After a swap, advance the pointer an extra step.
           * This is critical to ensure that the boy who just moved does not
           * move again in the same time step. It correctly simulates a
           * single, simultaneous swap across the whole queue per second.
           */
          i++;
        }
      }
    }
    cout << s << endl;
  }

  return 0;
}
