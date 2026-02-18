
/**

 * @file 266B.CPP

 * @brief Solution to the "Queue at the School" problem (266B) from Codeforces.

 * @details This program simulates a queue of boys ('B') and girls ('G'). Over 't'

 * seconds, any boy standing directly in front of a girl swaps places with her.

 *

 * @author Your Name (or leave blank)

 *

 * @algorithm The solution is a discrete-time simulation.

 * 1. Read the queue length (n), time steps (t), and the initial queue string (s).

 * 2. An outer loop runs for `t` iterations, one for each second.

 * 3. An inner `for` loop scans the queue to find all 'BG' pairs.

 * 4. When a 'BG' pair is found at `i` and `i+1`, their positions are swapped.

 * 5. The loop index `i` is then explicitly incremented inside the `if` block. This,

 *    along with the `for` loop's own increment, causes the loop to advance by two

 *    positions after a swap, correctly modeling that a child moves at most once

 *    per time step.

 * 6. After `t` seconds, the final state of the queue string is printed.

 *

 * @complexity

 * Time Complexity: O(T * N), where T is the number of time steps and N is the length of the queue.

 * The simulation involves T passes over the string of length N.

 * Space Complexity: O(N) to store the queue string.

 */

#include <bits/stdc++.h>



using namespace std;



int main() {

  int n, t;

  string s;

  cin >> n;

  cin >> t;

  cin >> s;

  // This `i` is shadowed by the loop variable in the `for` loop.

  int i = 0;

  /**

   * @block

   * @brief Simulates the swapping process over 't' seconds.

   */

  while (t--) {

    /**

     * @block

     * @brief A single pass through the queue for one time step.

     */

    for (int i = 0; i < n - 1; i++) {

      // If a Boy is in front of a Girl, they swap.

      if (s[i] == 'B' && s[i + 1] == 'G') {

        swap(s[i], s[i + 1]);

        // Manually advance the index to prevent the swapped 'B' from moving again

        // in the same time step. The for-loop's increment will also apply.

        ++i;

      }

    }

  }

  cout << s;

  return 0;

}
