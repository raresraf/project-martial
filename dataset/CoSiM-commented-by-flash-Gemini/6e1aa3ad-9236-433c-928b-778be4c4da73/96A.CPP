/**
 * @file 96A.CPP
 * @brief Solution to a competitive programming problem: "Football".
 * Determines if a "dangerous" situation occurs in a football match,
 * defined as seven consecutive players of the same team.
 * Algorithm: Linear scan to count consecutive identical characters.
 * Time Complexity: O(L) where L is the length of the input string (situation).
 * Space Complexity: O(1)
 */
// 56680198	Practice:havoccboi	96A - 10	GNU C++17	Accepted
// 62 ms	16 KB	2019-07-07 20:54:01	2019-07-07 20:54:01

#include <bits/stdc++.h>

using namespace std;

/**
 * @brief Main function to solve the Football problem.
 * Reads a string representing the football situation and checks for seven consecutive
 * identical characters to determine if it's "dangerous".
 */
int main() {
  string s;
  cin >> s;
  int c = 1; // Counter for consecutive identical characters.
  bool flag = false; // Flag to indicate if a dangerous situation is found.

  /**
   * Block Logic: Initial check for string length.
   * If the string is shorter than 7, a "dangerous" situation is impossible.
   * Pre-condition: 's' contains the input string.
   */
  if ((int)s.size() < 7) {
    cout << "NO" << endl;
  } else {
    /**
     * Block Logic: Iterates through the string to count consecutive identical characters.
     * Pre-condition: 's' has a length of at least 7. 'c' is initialized to 1.
     * Invariant: 'c' tracks the current streak of identical characters.
     */
    for (int i = 0; i < (int)s.size() - 1; i++) {
      /**
       * Block Logic: Increments counter if current character matches the next one.
       * Otherwise, resets the counter to 1 for a new streak.
       */
      if (s[i] == s[i + 1]) {
        c++;
      } else {
        c = 1;
      }

      /**
       * Block Logic: Checks if a dangerous situation (7 consecutive) has been reached.
       * If so, set the flag and break early as no further checking is needed.
       */
      if (c == 7) {
        flag = true;
        break;
      }
    }
    /**
     * Block Logic: Final determination after checking the entire string or finding a dangerous situation.
     * If 'flag' is true, a "dangerous" situation was found; otherwise, it was not.
     */
    if (flag == true)
      cout << "YES" << endl;
    else
      cout << "NO" << endl;
  }
}
