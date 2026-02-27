/**
 * @file 266B.CPP
 * @brief Solves the "Queue at the School" problem from Codeforces.
 * @details This program simulates the process of boys and girls swapping places
 * in a queue over a given time. In each time step, any boy standing
 * immediately in front of a girl swaps places with her.
 *
 * @algorithm
 * The simulation proceeds in discrete time steps.
 * 1. Read the queue size `n`, the total time `t`, and the initial queue `s`.
 * 2. Loop `t` times to simulate each second.
 * 3. Inside the time loop, perform a single pass through the queue.
 *    a. Iterate from the start of the queue.
 *    b. If a 'B' (boy) is found at index `i` and a 'G' (girl) at `i+1`, swap them.
 *    c. After a swap, increment the loop counter `i` an extra time. This is
 *       crucial because it prevents the newly moved boy from swapping again
 *       in the same time step, effectively modeling simultaneous swaps.
 * 4. After `t` seconds, print the final state of the queue.
 *
 * @complexity
 * Time Complexity: O(t * n), where t is the time and n is the queue length,
 * due to the nested loops.
 * Space Complexity: O(n) to store the string representing the queue.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  string s; // Represents the queue of children.
  int n;    // The number of children in the queue.
  int t;    // The time in seconds to simulate.
  
  cin >> n >> t >> s;
  
  /**
   * @block
   * @brief Outer loop to simulate the progression of time for 't' seconds.
   */
  while (t--) {
    /**
     * @block
     * @brief Inner loop to perform all swaps for a single time step.
     */
    for (int i = 0; i < n; i++) {
      // Find a Boy ('B') directly in front of a Girl ('G').
      if (s[i] == 'B' && s[i + 1] == 'G') {
        swap(s[i], s[i + 1]);
        
        // Inline: Crucial step to prevent a single 'B' from moving more than
        // one position per time step. This correctly models the simultaneous
        // nature of the swaps within one second.
        i++;
      }
    }
  }
  
  cout << s << endl;
  return 0;
}
