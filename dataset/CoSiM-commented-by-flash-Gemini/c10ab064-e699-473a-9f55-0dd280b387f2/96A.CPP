/**
 * @file 96A.CPP
 * @brief This program solves the "Football" problem (Codeforces 96A).
 *   It determines if a given string of '0's and '1's represents a "dangerous" situation.
 *   A situation is considered dangerous if there are 7 or more consecutive '0's
 *   OR 7 or more consecutive '1's in the string.
 *   Problem description: https://codeforces.com/problemset/problem/96/A
 * * Algorithm: Sliding window check.
 *   It iterates through the string using a window of 7 characters. For each window,
 *   it checks if all 7 characters within that window are identical. If such a sequence
 *   is found, the situation is dangerous.
 * Time Complexity: O(L) where L is the length of the input string.
 *   The `strlen(str)` in the loop condition is assumed to be optimized or evaluated once.
 * Space Complexity: O(L) for storing the input character array.
 */
#include <cstring>  // Required for `strlen`.
#include <iostream> // Required for `std::cin`, `std::cout`.

using namespace std;

/**
 * @brief Main function to check if a football situation is "dangerous".
 * Functional Utility: Reads a string of '0's and '1's, checks for 7 or more
 *                     consecutive identical characters using a sliding window,
 *                     and prints "YES" or "NO".
 */
int main() {
  int i, j, flag = 0; // `i`: loop counter, `j`: unused, `flag`: indicates if a dangerous situation is found.
  char str[110];      // C-style string to store the input. Max length 109 characters + null terminator.
  // Block Logic: Reads the input string `str`.
  cin >> str;

  // Conditional Logic: Checks if the string is long enough to contain 7 consecutive characters.
  // If the string length is less than 7, it's impossible to have 7 consecutive characters.
  if (strlen(str) >= 7) {
    /**
     * Block Logic: Iterates through the string with a sliding window of 7 characters.
     * The loop runs `strlen(str) - 6` times to ensure that `str[i+6]` is always within bounds.
     */
    for (i = 0; i < strlen(str) - 6; i++) {
      /**
       * Conditional Logic: Directly checks if 7 consecutive characters starting from `str[i]` are identical.
       * This checks if `str[i] == str[i+1] == str[i+2] == ... == str[i+6]`.
       */
      if ((str[i] == str[i + 1]) && (str[i + 1] == str[i + 2]) &&
          (str[i + 2] == str[i + 3]) && (str[i + 3] == str[i + 4]) &&
          (str[i + 4] == str[i + 5]) && (str[i + 5] == str[i + 6])) {
        flag = 1; // Sets the flag to indicate a dangerous situation.
        break;    // Breaks the loop early as the condition is met.
      }
    }
  }
  // Conditional Logic: Prints "YES" or "NO" based on the `flag`.
  if (flag == 1)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
  return 0;
}
