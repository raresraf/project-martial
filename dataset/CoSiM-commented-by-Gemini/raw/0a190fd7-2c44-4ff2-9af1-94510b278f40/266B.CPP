/**
 * @file 266B.CPP
 * @brief Solves the "Queue at the School" problem from Codeforces.
 * @details This program simulates the state of a queue of boys ('B') and girls ('G') over a given number of time steps 't'.
 * In each step, any boy standing immediately in front of a girl swaps places with her. This implementation
 * builds a new string for each time step.
 *
 * @see https://codeforces.com/problemset/problem/266/B
 *
 * Algorithm:
 * 1. Read the number of children (n), the time duration (t), and the initial queue arrangement (s).
 * 2. Loop 't' times to simulate the passage of time.
 * 3. In each time step, create a new empty string `s2`.
 * 4. Iterate through the current queue string `s`.
 * 5. If a 'B' is found before a 'G', append "GB" to `s2` and skip the next character in `s`.
 * 6. Otherwise, append the current character to `s2`.
 * 7. After iterating through `s`, replace `s` with the newly constructed string `s2`.
 * 8. After 't' steps, print the final queue arrangement.
 *
 * Time Complexity: O(t * n), where 't' is the number of time steps and 'n' is the length of the queue.
 * Space Complexity: O(n) to store the queue string and the intermediate string `s2` in each step.
 */

#include <bits/stdc++.h>

using namespace std;

/**
 * @brief Simulates the queue swapping process for 't' time steps.
 * @param n The number of children in the queue.
 * @param t The number of time steps to simulate.
 * @param s The initial arrangement of the queue as a string.
 * @return The final arrangement of the queue after 't' seconds.
 */
string solve(int n, int t, string s) {
  /**
   * Block Logic: Orchestrates the temporal progression of the simulation.
   * Invariant: `s` holds the state of the queue at the beginning of each time step.
   */
  while (t--) {
    int i = 0;
    string s2 = "";
    /**
     * Block Logic: Scans the current queue `s` to build the next state `s2`.
     * This pass identifies all 'BG' pairs and swaps them in the new string.
     */
    while (i < s.length()) {
      // Pre-condition: Check for an adjacent 'B' and 'G' pair within bounds.
      if (i + 1 < s.length() && s[i] == 'B' && s[i + 1] == 'G') {
        // Perform the swap in the new string `s2`.
        s2 += "GB";
        // Advance the index by 2 as two characters have been processed.
        i += 2;
      } else {
        // If no swap occurs, copy the character as is.
        s2 += s[i];
        i++;
      }
    }
    // Update the queue state for the next time step.
    s = s2;
  }

  return s;
}

int main() {
  /**
   * Functional Utility: Decouples C++ streams from standard C streams and unties cin from cout
   * to optimize I/O performance, which is a common practice in competitive programming.
   */
  ios::sync_with_stdio(false);
  cin.tie(NULL);

  int n, t;
  cin >> n >> t;
  string s;
  cin >> s;
  cout << solve(n, t, s) << endl;
}