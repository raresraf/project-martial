
/******************************************
 *  Author : Debopriya Deb Roy
 *  Created On : Wed Jul 01 2020
 *  File : c++ file
 *******************************************/

#include <bits/stdc++.h>
using namespace std;

/**
 * @file 96A.CPP
 * @brief Solution to Codeforces problem 96A, "Football".
 * @details This program determines if a situation is "dangerous" in a football
 * match, defined as having at least seven consecutive players of the same team.
 *
 * Algorithm:
 * The program iterates through the input string, which represents a line of
 * players. It maintains a `check` variable to store the character of the
 * current run (e.g., '0' or '1') and a `count` variable for the length of
 * this run. It compares each character to the `check` character. If they match,
 * `count` is incremented. If they don't, a new run has started, so `check` is
 * updated and `count` is reset to 1. The loop terminates immediately if a
 * run of 7 is found.
 *
 * Time Complexity: O(N), where N is the length of the string, due to the
 * single pass. The loop may exit early if a dangerous situation is found.
 * Space Complexity: O(N) to store the input string.
 */
int main() {
  // `i` is the loop index, other variables are unused.
  int i = 1, j, k, l, m, n;
  string s;
  cin >> s;

  int size = s.size();
  // `count`: tracks the length of the current consecutive run.
  // `check`: stores the character of the current run.
  int count = 1, check = s[0];

  /**
   * @brief Main loop to find a dangerous situation.
   * This loop iterates from the second character to the end of the string.
   */
  while (size--) {
    // Compare the current character with the character of the ongoing run.
    if (check == s[i]) {
      count++;
      // If a run of 7 is found, no need to check further.
      if (count == 7)
        break;
    } else {
      // The run is broken; reset the count and start a new run.
      count = 1;
      check = s[i];
    }
    // Manually increment the index for the next character.
    i++;
  }

  // Final check to see if a run of 7 was ever reached.
  if (count == 7)
    cout << "YES";
  else {
    cout << "NO";
  }

  return 0;
}
