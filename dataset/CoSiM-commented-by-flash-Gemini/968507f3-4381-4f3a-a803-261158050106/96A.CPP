/**
 * @file 96A.CPP
 * @brief Determines if a football player is "dangerous" based on a sequence of goals and misses.
 * * Algorithm: Sequential Scan for Consecutive Identical Characters.
 * Time Complexity: $O(N)$ where N is the length of the input string, due to a single pass.
 * Space Complexity: $O(1)$ for storing counters.
 *
 * This program analyzes a binary string (representing a player's performance, e.g., '0' for miss, '1' for goal).
 * A player is considered "dangerous" if there are 7 or more consecutive identical characters (either '0's or '1's).
 */

#include <bits/stdc++.h> // Includes various standard libraries for competitive programming.
using namespace std;

int main() {
  string s; // The input binary string representing player performance.
  cin >> s; // Reads the binary string from standard input.

  // Variables to track consecutive counts and maximum consecutive counts.
  int count_0 = 0, count_1 = 0; // Current consecutive count of '0's and '1's.
  int max_0 = 0, max_1 = 0;     // Maximum consecutive count of '0's and '1's encountered so far.

  /**
   * Block Logic: Iterates through the input string to find the maximum number of consecutive '0's and '1's.
   * Pre-condition: `s` contains a binary string.
   * Invariant: `max_0` and `max_1` always store the highest consecutive counts found up to the current character.
   */
  for (int i = 0; i < s.length(); i++) {
    if (s[i] == '0') {
      // Inline: If '0' is encountered, reset the '1' counter and increment the '0' counter.
      count_1 = 0;
      count_0++;
      // Inline: Update `max_0` if the current `count_0` is greater.
      if (max_0 < count_0)
        max_0 = count_0;
    } else { // s[i] == '1'
      // Inline: If '1' is encountered, reset the '0' counter and increment the '1' counter.
      count_0 = 0;
      count_1++;
      // Inline: Update `max_1` if the current `count_1` is greater.
      if (max_1 < count_1)
        max_1 = count_1;
    }
  }

  /**
   * Block Logic: Determines if the player is "dangerous" based on the maximum consecutive counts.
   * Pre-condition: `max_0` and `max_1` hold the maximum consecutive counts from the string.
   * Post-condition: Prints "YES" if either maximum count is 7 or more, "NO" otherwise.
   */
  if ((max_0 >= 7) || (max_1 >= 7))
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0; // Standard successful exit code for main function.
}
