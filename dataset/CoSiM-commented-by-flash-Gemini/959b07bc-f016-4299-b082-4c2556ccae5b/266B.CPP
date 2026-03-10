/**
 * @file 266B.CPP
 * @brief This program simulates a queue of boys ('B') and girls ('G') where adjacent 'B' and 'G' pairs swap positions over time.
 *   The simulation runs for a given number of seconds, with all possible swaps happening simultaneously in each second.
 *   Problem description: "Queue at the School", Codeforces 266B.
 * * Algorithm: Iterative simulation of swaps. For each second, the queue is traversed to identify and execute 'G' and 'B' swaps.
 *   The inner loop's advancement mechanism ensures that a character who moves does not immediately participate in another swap
 *   in the same time step, correctly simulating simultaneous movements.
 * Time Complexity: O(t * n) where 't' is the number of time steps (seconds) and 'n' is the length of the queue.
 *   Given typical constraints (n <= 50, t <= 50), this is very efficient (e.g., 50*50 = 2500 operations).
 * Space Complexity: O(n) to store the string representing the queue order.
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries.
using namespace std;

/**
 * @brief Main function to simulate the queue rearrangement.
 * Functional Utility: Reads the queue length, simulation time, and initial queue order.
 *                     Simulates the swaps over time and prints the final queue order.
 */
int main() {
  int n, t;    // `n` for queue length, `t` for number of seconds.
  string s;    // Stores the current order of the queue.

  // Block Logic: Reads the queue length `n`, the number of seconds `t`, and the initial queue string `s`.
  cin >> n >> t >> s;
  /**
   * Block Logic: Simulates the passage of time. The outer `while` loop executes for `t` iterations,
   *              each iteration representing one second of time.
   * Pre-condition: `t` is a non-negative integer representing the total seconds.
   * Invariant: After each iteration, the queue reflects the state after one second of swaps.
   * Inline: `t--` decrements the counter and evaluates to true as long as `t` is greater than 0.
   */
  while (t--) {
    /**
     * Block Logic: Traverses the queue to identify and perform 'G' and 'B' swaps.
     * Pre-condition: `s` contains the current queue configuration.
     * Invariant: After each iteration of the inner loop, any 'G' followed by 'B' has swapped,
     *            and `i` is advanced to correctly handle simultaneous swaps.
     */
    for (int i = 1; i < n; i++) { // Iterates from the second character to the end.
      /**
       * Block Logic: Checks if a girl ('G') is at `s[i]` and a boy ('B') is at `s[i-1]`.
       * This represents a 'B' followed by 'G' in the queue (relative to the front).
       * If true, they swap positions.
       */
      if (s[i] == 'G' && s[i - 1] == 'B') {
        s[i] = 'B';     // The boy moves to the girl's original position.
        s[i - 1] = 'G'; // The girl moves to the boy's original position.
        // Inline: Increments 'i' to skip the newly moved characters. This prevents a character
        // from moving multiple positions in a single second, correctly simulating
        // simultaneous swaps for all pairs.
        ++i;
      }
    }
  }
  // Block Logic: Prints the final arrangement of the queue after all seconds have passed.
  cout << s << endl;
  return 0;
}