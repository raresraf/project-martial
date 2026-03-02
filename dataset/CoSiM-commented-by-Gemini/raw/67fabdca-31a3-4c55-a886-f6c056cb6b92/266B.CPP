#include <bits/stdc++.h>
using namespace std;

/**
 * @file 266B.CPP
 * @brief Solution to the "Queue at the School" problem from Codeforces (Problem 266B).
 * @details This program simulates a queue of boys ('B') and girls ('G'). Over `t` seconds,
 * any boy standing immediately in front of a girl swaps places with her. This process
 * is repeated for a given amount of time.
 *
 * Algorithm:
 * The simulation is performed using nested loops. The outer loop iterates `t` times,
 * representing each second of time passed. The inner loop iterates through the queue
 * from left to right. Inside the inner loop, it checks for adjacent pairs of a boy
 * followed by a girl ('BG'). When such a pair is found, their positions are swapped.
 * After a swap, the inner loop's index `j` is incremented an extra time. This is a
 * crucial step to ensure that in a single time step, a child moves at most one position,
 * and the scan continues after the swapped pair.
 *
 * Time Complexity: O(n * t), where `n` is the length of the queue and `t` is the
 * number of seconds, due to the nested loops.
 * Space Complexity: O(n) to store the string representing the queue.
 */
int main() {
  int n, t;
  string s;
  cin >> n >> t >> s;

  /**
   * @brief Outer loop for simulating the passage of time.
   * Runs for `t` seconds.
   */
  for (int i = 0; i < t; i++) {
    /**
     * @brief Inner loop to scan the queue and perform swaps for the current second.
     */
    for (int j = 0; j < n; ++j) {
      // Check for a boy 'B' followed by a girl 'G'.
      if (s[j] == 'B' && s[j + 1] == 'G') {
        // Swap their positions.
        swap(s[j], s[j + 1]);
        // Increment `j` to move past the swapped pair. This prevents
        // a single person from moving multiple times in one second.
        ++j;
      }
    }
  }
  // Print the final state of the queue.
  cout << s;
  return 0;
}
