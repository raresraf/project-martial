/**
 * @file 96A.CPP
 * @brief Solves the "Football" problem on Codeforces (Problem 96A).
 * @details This program determines if a situation in a football match is "dangerous".
 * A situation is dangerous if there are 7 or more consecutive players of the same team.
 * The input is a string of '0's and '1's representing players from two different teams.
 *
 * Algorithm: Single-pass consecutive character counting.
 * Time Complexity: O(N), where N is the length of the input string, as it requires a single traversal.
 * Space Complexity: O(N) to store the input string. The maximum length is bounded by the problem statement (typically 100).
 */
#include <bits/stdc++.h>
#include <string>

using namespace std;
int main() {
  string str;
  // Reads the player sequence. `setw(101)` is used as a safeguard to limit
  // input to 100 characters plus a null terminator, preventing buffer overflows,
  // aligned with typical competitive programming constraints for this problem.
  cin >> setw(101) >> str;
  
  // `flag` indicates if a dangerous situation has been found.
  // `one` and `zero` count consecutive occurrences of '1's and '0's respectively.
  int flag = 0, one(0), zero(0);

  /**
   * @brief Block Logic: Iterates through the string to find consecutive players.
   * Invariant: At each step `i`, `one` or `zero` holds the count of identical
   * consecutive characters ending at `str[i]`.
   */
  for (int i = 0; i < str.size(); i++) {

    // Pre-condition: Checks the character at the current position.
    if (str[i] == '1') {
      one++;
      // A '1' breaks any sequence of '0's, so reset the `zero` counter.
      if (zero != 0) {
        zero = 0;
      }
    } else if (str[i] == '0') {
      zero++;
      // A '0' breaks any sequence of '1's, so reset the `one` counter.
      if (one != 0) {
        one = 0;
      }
    }

    // Pre-condition: Checks if either counter has reached the "dangerous" threshold of 7.
    if (one == 7 || zero == 7) {
      cout << "YES" << endl;
      flag = 1; // Mark that a dangerous situation was found.
      break;    // Exit the loop as the condition is met.
    }
  }
  
  // If the loop completes without the flag being set, no sequence of 7 was found.
  if (flag == 0) {
    cout << "NO" << endl;
  }
}
