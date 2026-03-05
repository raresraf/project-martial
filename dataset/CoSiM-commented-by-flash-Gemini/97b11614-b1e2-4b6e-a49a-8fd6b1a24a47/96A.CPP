/**
 * @file 96A.CPP
 * @brief Solution to a problem that detects "dangerous" football positions,
 *        defined as a sequence of 7 or more consecutive players of the same team.
 *        The program reads a string representing player teams and outputs "YES"
 *        if a dangerous position is found, otherwise "NO".
 *
 * Algorithm: Iterative scan with a counter for consecutive identical characters.
 *            The algorithm iterates through the input string, maintaining a count
 *            of consecutive identical characters. If the count reaches 7, it
 *            immediately determines a "dangerous" position exists.
 * Time Complexity: O(N), where N is the length of the input string.
 *                  The string is traversed once.
 * Space Complexity: O(1), excluding input storage.
 */

#include <bits/stdc++.h>
using namespace std;
int main() {
  // Functional Utility: Initializes a counter for consecutive identical characters
  // and a flag to indicate if a "dangerous" position has been found.
  int c = 1, i, k = 0;
  string s;
  // Block Logic: Reads the input string representing player teams.
  cin >> s;
  /**
   * Block Logic: Iterates through the string to detect sequences of 7 or more
   *             consecutive identical characters.
   * Invariant: `c` holds the count of consecutive identical characters ending at `s[i-1]`.
   *            `k` is 1 if a dangerous position is found, 0 otherwise.
   */
  for (i = 1; i < s.length(); i++) {
    // Pre-condition: Checks if the current character is the same as the previous one.
    if (s[i] == s[i - 1]) {
      // Inline: Increments the counter for consecutive identical characters.
      c++;

      // Post-condition: If 7 consecutive identical characters are found,
      // prints "YES", sets the flag, and terminates the loop.
      if (c == 7) {
        cout << "YES" << endl;
        k = 1;
        break;
      }
    } else {
      // Block Logic: Resets the counter if the current character is different
      // from the previous one, starting a new sequence.
      c = 1;
    }
  }
  // Post-condition: If the loop completes without finding 7 consecutive identical characters,
  // prints "NO".
  if (k == 0) {
    cout << "NO" << endl;
  }
  return 0;
}
