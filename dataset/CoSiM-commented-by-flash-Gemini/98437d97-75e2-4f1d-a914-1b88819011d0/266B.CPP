/**
 * @file 266B.CPP
 * @brief This program simulates a queue reordering process where boys ('B') move
 * forward if they are directly followed by a girl ('G'). The simulation runs
 * for a specified number of time steps.
 *
 * Algorithm: Iterative simulation with local swaps.
 * 1. Read the length of the queue (n) and the number of simulation steps (t).
 * 2. Read the initial configuration of the queue as a string.
 * 3. For 't' time steps, iterate through the queue:
 *    a. If a 'B' is followed by a 'G', swap their positions.
 *    b. Skip the next position to ensure a 'B' moves only one step per second.
 * Time Complexity: O(T * N) where T is the number of time steps and N is the length of the queue.
 * Space Complexity: O(N) to store the queue string.
 */

#include <bits/stdc++.h>
using namespace std;
#define ll long long
#define ff first
#define ss second
#define pb push_back
#define mod 1000000007
#define pi acos(-1)

/**
 * @brief Main function of the program.
 * Initializes the queue simulation, processes the reordering, and prints the final state.
 * @pre `n` and `t` are positive integers; `s` consists of 'B' and 'G' characters.
 * @post Prints the final arrangement of the queue after `t` time steps.
 * @return 0 upon successful execution.
 */
int main() {
  // freopen("input.txt", "r", stdin); // Original commented line, preserved.
  // freopen("output.txt", "w", stdout); // Original commented line, preserved.

  // Functional Utility: Optimizes C++ standard streams for faster input/output, common in competitive programming.
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

  // Functional Utility: 'n' for queue length, 't' for simulation time steps, 's' for the queue string.
  int n, t;
  string s;

  // Functional Utility: Reads the queue length and simulation steps.
  cin >> n >> t;
  // Functional Utility: Reads the initial queue configuration.
  cin >> s;

  // Block Logic: Controls the simulation over 't' discrete time steps.
  // Invariant: At the beginning of each iteration, 's' represents the queue state after (T_initial - t) seconds.
  while (t--) {
    // Block Logic: Performs a single pass through the queue to apply all possible swaps for the current time step.
    // Invariant: By the end of this loop, all boys immediately followed by girls have moved one step forward.
    for (int i = 0; i < s.length(); i++) {
      // Block Logic: Checks for an adjacent 'B' (boy) followed by a 'G' (girl) to trigger a swap.
      // Pre-condition: `i` is a valid index within the current string length such that `i+1` is also valid.
      // Note: The original loop condition `i < s.length()` can lead to out-of-bounds access for `s[i+1]`
      // when `i` is `s.length() - 1`. For strict compliance with "Zero Code Mutation", this is left as is.
      if (s[i] == 'B' && s[i + 1] == 'G') {
        // Functional Utility: Swaps the positions of the boy and the girl.
        swap(s[i], s[i + 1]);
        // Inline: Increments 'i' to skip the newly moved girl, preventing the same boy from moving twice in one second.
        i++;
      }
    }
  }

  // Functional Utility: Prints the final state of the queue after all simulation steps are complete.
  cout << s << "
";

  return 0;
}
