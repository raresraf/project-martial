/**
 * @file 96A.CPP
 * @brief Solution for Codeforces Problem 96A, determining if a football game situation is "dangerous".
 * @details The problem defines a "dangerous" situation as one where seven or more players of the same team are in a row. The input is a string of '0's and '1's representing players from two teams.
 *
 * @algorithm Linear Scan
 * @description This program scans the input string to find if there are 7 consecutive identical characters. It uses a counter that increments for same consecutive characters and resets otherwise.
 * @complexity
 * - Time: O(N), where N is the length of the input string, as it requires a single pass.
 * - Space: O(1), as it uses a constant amount of extra space.
 */
#include <iostream>
using namespace std;

int main() {
  string s;
  getline(cin, s);
  string out = "NO";

  /**
   * @brief Pre-check to avoid processing strings that are too short to be dangerous.
   * A dangerous situation requires at least 7 players, so strings shorter than that are immediately deemed safe.
   */
  if (s.length() >= 7) {
    int c = 1;
    /**
     * @brief Scans the string to find a sequence of 7 identical characters.
     * @invariant `c` holds the length of the current consecutive sequence of identical characters ending at `s[i]`.
     */
    for (int i = 1; i < s.length(); i++) {
      if (s[i] == s[i - 1])
        c++;
      else
        c = 1;
      
      // If a sequence of 7 is found, the situation is dangerous.
      if (c == 7) {
        out = "YES";
        break;
      }
    }
  }
  cout << out;
  return 0;
}
