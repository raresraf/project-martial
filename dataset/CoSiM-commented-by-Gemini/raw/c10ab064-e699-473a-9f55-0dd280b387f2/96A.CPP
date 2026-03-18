/**
 * @file 96A.CPP
 * @brief Solution for a string analysis problem, likely from a competitive programming context.
 * @details This program determines if there is a sequence of 7 or more identical
 *          consecutive characters within a given input string.
 *
 * Algorithm:
 *   - The program employs a single-pass linear scan through the input string.
 *   - It uses a fixed-size sliding window of 7 characters.
 *   - For each window, it checks if all characters are identical.
 *
 * Time Complexity: O(N), where N is the length of the input string. The program
 * performs a single pass over the string.
 *
 * Space Complexity: O(L), where L is the maximum length of the string (110 in
 * this case), to store the input string itself. The auxiliary space complexity
 * is O(1).
 */

#include <cstring>
#include <iostream>

using namespace std;
int main() {
  int i, j, flag = 0;
  char str[110];
  cin >> str;

  /**
   * Pre-condition: A sequence of 7 identical characters can only exist if the
   * string has at least 7 characters. This check avoids unnecessary iteration.
   */
  if (strlen(str) >= 7) {
    /**
     * Block Logic: Iterates through the string with a sliding window approach.
     * The loop stops at `strlen(str) - 6` to ensure the window of 7
     * characters does not read past the end of the string.
     */
    for (i = 0; i < strlen(str) - 6; i++) {
      /**
       * Block Logic: Checks for a dangerous situation where a player is surrounded
       * by 7 or more identical players (characters). It compares 7 consecutive
       * characters to see if they are all the same.
       */
      if ((str[i] == str[i + 1]) && (str[i + 1] == str[i + 2]) &&
          (str[i + 2] == str[i + 3]) && (str[i + 3] == str[i + 4]) &&
          (str[i + 4] == str[i + 5]) && (str[i + 5] == str[i + 6])) {
        flag = 1;
        break; // Exit early as the condition has been met.
      }
    }
  }

  // Output: Reports whether a dangerous situation was found.
  if (flag == 1)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
  return 0;
}
