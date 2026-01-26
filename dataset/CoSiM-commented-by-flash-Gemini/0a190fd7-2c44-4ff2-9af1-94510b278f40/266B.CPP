/**
 * @file 266B.CPP
 * @brief Solution for a competitive programming problem simulating a queue reordering process.
 * Algorithm: Iterative simulation of 'Boy' (B) and 'Girl' (G) swaps in a queue over time.
 *            In each time step, adjacent 'BG' pairs are swapped to 'GB'.
 * Time Complexity: O(T * N) where T is the number of time steps and N is the length of the string (queue).
 * Space Complexity: O(N) to store the intermediate string state.
 */

#include <bits/stdc++.h>

using namespace std;

/**
 * @brief Simulates the reordering process in a queue.
 *
 * This function takes an initial queue state represented as a string `s`,
 * and simulates `t` time steps. In each step, if a 'B' (Boy) is immediately
 * followed by a 'G' (Girl), they swap positions.
 *
 * @param n The length of the queue (string).
 * @param t The number of time steps to simulate.
 * @param s The initial state of the queue as a string.
 * @return The final state of the queue after `t` time steps.
 */
string solve(int n, int t, string s) {
  // Block Logic: Orchestrates the temporal progression of the simulation.
  // Invariant: At the start of each iteration, the string `s` represents the queue state at time T-k.
  while (t--) {
    int i = 0; // i: Index for iterating through the queue string.
    string s2 = ""; // s2: Temporary string to build the next state of the queue.
    // Block Logic: Performs a single-pass sweep to resolve priority inversions.
    // Logic: Identifies 'B' (Boy) and 'G' (Girl) pairs where 'B' precedes 'G'
    // and performs a local state transition (swap) in the `s2` string.
    while (i < s.length()) {
      // Block Logic: Checks for an adjacent 'BG' pair to perform a swap.
      if (i + 1 < s.length() && s[i] == 'B' && s[i + 1] == 'G') {
        s2 += "GB"; // Perform swap: 'BG' becomes 'GB'.
        i += 2;     // Skip the next character as it was part of the swap.
      } else {
        s2 += s[i]; // Append the current character as no swap occurred.
        i++;        // Move to the next character.
      }
    }
    s = s2; // Update the queue state for the next time step.
  }

  return s; // Return the final state of the queue.
}

/**
 * @brief Main function to read input, call the solver, and print the result.
 *
 * @param None
 * @return 0 on successful execution.
 */
int main() {
  // Functional Utility: Decouples C++ streams from standard C streams and unties cin from cout
  // to optimize high-throughput I/O required for competitive programming constraints.
  ios::sync_with_stdio(false);
  cin.tie(NULL);

  // n: The length of the queue (string).
  // t: The number of time steps to simulate.
  int n, t;
  cin >> n >> t;
  // s: The initial state of the queue as a string.
  string s;
  cin >> s;
  cout << solve(n, t, s) << endl; // Call the solver and print the final queue state.
}
