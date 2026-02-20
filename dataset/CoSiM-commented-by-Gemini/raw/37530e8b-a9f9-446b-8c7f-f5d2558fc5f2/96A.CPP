/**
 * @file 96A.CPP
 * @brief Solves the "Football" problem by checking for consecutive players.
 * @author SHEHANUK AHSAN
 *
 * This program determines if a situation is "dangerous" in a line of football
 * players, which occurs if there are at least seven consecutive players of the
 * same team. The line is represented by a string of '0's and '1's.
 *
 * This implementation uses a brute-force approach, checking every possible
 * substring of length 7.
 *
 * @note There is a bug in this implementation. The initial check `strlen(a) > 7`
 * should be `strlen(a) >= 7` to correctly handle cases where the string has
 * exactly 7 identical characters.
 *
 * Time Complexity: O(L), where L is the length of the string. The loop runs
 * approximately L times, and the check inside is constant time.
 * Space Complexity: O(L_max) for the character array buffer.
 */
// made by : SHEHANUK AHSAN
//           3rd batch, Dept of CSE
//           BAUAT;

#include <cstdio>
#include <iostream>
#include <string.h>
using namespace std;

main() {
  int n, i, j, k; // n, j, k are declared but not used.
  char a[100];
  cin >> a;

  /**
   * Block Logic: Main check for a dangerous situation.
   * @note Bug: The condition should be `strlen(a) >= 7`. As written, it will
   * fail to detect a dangerous situation in a string of exactly 7 characters.
   */
  if (strlen(a) > 7) {
    /**
     * Block Logic: Brute-force scan.
     * This loop iterates through the string to find a sequence of 7
     * identical characters. The loop bound `strlen(a) - 6` is correct
     * to ensure the last check `a[i+6]` is valid.
     */
    for (i = 0; i < (strlen(a) - 6); i++) {
      // This hardcoded check verifies if 7 consecutive characters are all identical.
      if (a[i] == a[i + 1] && a[i] == a[i + 2] && a[i] == a[i + 3] &&
          a[i] == a[i + 4] && a[i] == a[i + 5] && a[i] == a[i + 6]) {
        cout << "YES";
        return 0; // Situation found, exit immediately.
      }
    }
  }

  // If the loop completes without finding a dangerous sequence, or if the
  // initial length check failed, the situation is not dangerous.
  cout << "NO";

  return 0;
}
