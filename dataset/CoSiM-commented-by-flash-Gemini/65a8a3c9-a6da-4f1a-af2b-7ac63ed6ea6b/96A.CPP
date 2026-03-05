/**
 * @file 96A.CPP
 * @brief Solution to the "Football" problem, also known as "Dangerous Football".
 *        The program determines if there is a "dangerous situation" in a sequence
 *        of players, which is defined as having 7 or more consecutive players
 *        from the same team (represented by '0' or '1').
 *
 * Algorithm: Iterative scan with separate counters for consecutive '0's and '1's.
 *            The algorithm iterates through the input string, maintaining two
 *            counters: one for consecutive '1's and one for consecutive '0's.
 *            If either counter reaches 7, a dangerous situation is declared.
 * Time Complexity: O(N), where N is the length of the input string.
 *                  The string is traversed once.
 * Space Complexity: O(1), excluding input string storage.
 */

#include <bits/stdc++.h>
#include <string>

using namespace std;

int main() {
  string str;
  // Functional Utility: Reads the input string `str`. `setw(101)` sets the maximum
  // width of the input to 100 characters plus a null terminator, preventing buffer overflows.
  cin >> setw(101) >> str;
  // Functional Utility: `flag` indicates if a dangerous situation is found (1 for yes, 0 for no).
  // `one` counts consecutive '1's, `zero` counts consecutive '0's.
  int flag = 0, one(0), zero(0);
  /**
   * Block Logic: Iterates through the input string to count consecutive identical characters.
   * Invariant: `one` stores the current count of consecutive '1's ending at `str[i]`,
   *            and `zero` stores the current count of consecutive '0's ending at `str[i]`.
   */
  for (int i = 0; i < str.size(); i++) {
    // Pre-condition: Checks if the current character is '1'.
    if (str[i] == '1') {
      // Inline: Increments the '1' counter.
      one++;
      // Block Logic: Resets the '0' counter if a '1' is encountered,
      // as the sequence of '0's is broken.
      if (zero != 0) {
        zero = 0;
      }
    }
    // Pre-condition: Checks if the current character is '0'.
    else if (str[i] == '0') {
      // Inline: Increments the '0' counter.
      zero++;
      // Block Logic: Resets the '1' counter if a '0' is encountered,
      // as the sequence of '1's is broken.
      if (one != 0) {
        one = 0;
      }
    }
    // Post-condition: Checks if either counter has reached 7 or more.
    if (one == 7 || zero == 7) {
      // Block Logic: If a dangerous situation is found, prints "YES",
      // sets the flag, and breaks the loop.
      cout << "YES" << endl;
      flag = 1;
      break;
    }
  }
  // Post-condition: After the loop, if no dangerous situation was found (`flag` is still 0),
  // prints "NO".
  if (flag == 0) {
    cout << "NO" << endl;
  }
  return 0;
}
