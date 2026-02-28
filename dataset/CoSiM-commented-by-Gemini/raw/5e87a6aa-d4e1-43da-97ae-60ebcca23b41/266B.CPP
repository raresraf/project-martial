/**
 * @file 266B.CPP
 * @brief An incorrect solution for the "Queue at the School" problem from Codeforces.
 * @details This program attempts to simulate the changing positions in a queue of boys ('B')
 * and girls ('G'). However, the simulation logic is flawed, leading to incorrect results.
 *
 * @algorithm Flawed Discrete Time Simulation
 * The core logic attempts to simulate the queue swaps over `t` seconds.
 * 1. An outer loop runs `t` times for each second.
 * 2. An inner `for` loop scans the queue for adjacent 'BG' pairs.
 * 3. When a 'BG' pair at index `j` is found and swapped, the code executes `j++`.
 *
 * @flaw The critical flaw is in the `j++` statement after a swap. By only incrementing
 * `j` by one, a boy who has just moved from `j` to `j+1` can immediately be considered
 * for another swap with a girl at `j+2` in the very same time step. This allows a
 * single boy to move multiple places in one second, which violates the problem's rules.
 * A correct implementation would need to ensure that each child moves at most once
 * per time step, for instance by using a `while` loop and incrementing the index by 2
 * after a swap.
 *
 * @complexity
 * - Time: O(T * N), where T is the time and N is the length of the queue.
 * - Space: O(N), to store the queue string.
 */
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
  // n: the number of children in the queue.
  // t: the total time in seconds to simulate.
  int n, t;
  // Using C-style scanf for input.
  scanf("%d%d", &n, &t);
  
  // The string representing the queue.
  string s;
  cin >> s;

  /**
   * @brief Outer loop to simulate the passage of time for `t` seconds.
   */
  for (int i = 1; i <= t; i++) {
    /**
     * @brief Inner loop to scan the queue. This loop contains the logical flaw.
     */
    for (int j = 0; j < n - 1; j++) {
      // Pre-condition: Check if a boy is standing directly in front of a girl.
      if (s[j] == 'B' && s[j + 1] == 'G') {
        // Swap their positions.
        s[j + 1] = 'B';
        s[j] = 'G';
        
        // FLAW: Incrementing `j` by 1 here is incorrect.
        // The boy who moved to `j+1` can now interact with the person at `j+2`
        // in the same time step, leading to a chain reaction that violates the rules.
        j++;
      }
    }
  }

  // Prints the final, but likely incorrect, arrangement of the queue.
  cout << s << "\n";
  
  return 0; // Indicate successful execution.
}
