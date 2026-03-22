/**
 * @file 96A.CPP
 * @brief Solution to Codeforces problem 96A - Football.
 *        This program analyzes a sequence of player positions (represented by '0's and '1's)
 *        to determine if a "dangerous situation" exists. A dangerous situation is defined as
 *        seven or more players of the same team standing side by side.
 *
 * Algorithm:
 *   1. Reads a string `s` representing the sequence of players.
 *   2. Initializes a counter `cnt` and a boolean flag `flag`.
 *   3. First Pass: Iterates through the string to count consecutive '0's.
 *      - If a '0' is encountered, `cnt` is incremented.
 *      - If a '1' is encountered, `cnt` is reset to 0.
 *      - If `cnt` reaches 7 or more, `flag` is set to true.
 *   4. Second Pass: Resets `cnt` to 0. Iterates through the string again to count consecutive '1's.
 *      - If a '1' is encountered, `cnt` is incremented.
 *      - If a '0' is encountered, `cnt` is reset to 0.
 *      - If `cnt` reaches 7 or more, `flag` is set to true.
 *   5. Finally, if `flag` is true, prints "YES" (dangerous situation); otherwise, prints "NO".
 *
 * Time Complexity: O(L), where L is the length of the input string `s`, as it involves two linear passes.
 * Space Complexity: O(L) for storing the input string, plus O(1) for auxiliary variables.
 */

#include <iostream>
using namespace std;

int main() {
  /**
   * @brief Main function that solves the "Football" problem.
   *
   * Reads a string of '0's and '1's representing player positions and determines
   * if there are 7 or more consecutive players of the same team.
   *
   * @return int Returns 0 upon successful execution.
   */
  string s;
  cin >> s;
  int cnt = 0, sz = s.length();
  bool flag = false;
  /**
   * Block Logic: First pass to check for 7 or more consecutive '0's.
   * Invariant: `cnt` stores the current count of consecutive '0's.
   * Pre-condition: `s` contains the player sequence. `flag` is initially false.
   */
  for (int i = 0; i < sz; i++) {
    if (s[i] == '0')
      cnt++;
    else {
      cnt = 0;
    }
    // Conditional Logic: Checks if the current sequence of '0's is dangerous.
    if (cnt >= 7)
      flag = true;
  }
  cnt = 0; // Reset counter for the second pass.
  /**
   * Block Logic: Second pass to check for 7 or more consecutive '1's.
   * Invariant: `cnt` stores the current count of consecutive '1's.
   * Pre-condition: `s` contains the player sequence. `flag` may already be true from the first pass.
   */
  for (int i = 0; i < sz; i++) {
    if (s[i] == '1')
      cnt++;
    else {
      cnt = 0;
    }
    // Conditional Logic: Checks if the current sequence of '1's is dangerous.
    if (cnt >= 7)
      flag = true;
  }
  /**
   * Block Logic: Outputs the final determination of a dangerous situation.
   * Pre-condition: `flag` is true if any dangerous sequence (7+ '0's or 7+ '1's) was found.
   * Post-condition: "YES" is printed if `flag` is true, "NO" otherwise.
   */
  if (flag)
    cout << "YES";
  else
    cout << "NO";
  return 0;
}
