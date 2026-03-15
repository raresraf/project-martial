/**
 * @file 339A.CPP
 * @brief Sorts digits in a string representation of a sum.
 * @details This program takes a string representing a sum of single-digit numbers (e.g., "1+3+2"),
 * sorts the digits in non-decreasing order, and prints the new sum.
 * The sorting is performed on the characters of the string directly.
 *
 * @algorithm
 * 1. Reads the input string.
 * 2. Implements a selection sort-like algorithm to sort all characters in the string,
 *    including digits and '+' signs, based on their ASCII values. This results in
 *    '+' characters being grouped at the beginning.
 * 3. Iterates through the sorted string, printing only the digit characters separated by '+'.
 *
 * @complexity
 * - Time: O(L^2), where L is the length of the string, due to the nested loops for sorting.
 * - Space: O(L) to store the input string.
 */
#include <cstring>
#include <iostream>
using namespace std;
int main() {
  string s;
  cin >> s;
  int l = s.size();

  /**
   * @brief In-place character sort block.
   * This nested loop implements a selection sort-like algorithm to arrange all characters
   * (digits and '+') in the string `s` in ascending order of their ASCII values.
   */
  for (int i = 0; i < l; i++) {
    for (int j = i + 1; j < l; j++) {
      if (s[i] > s[j]) {
        // Inline: Swaps characters s[i] and s[j] using arithmetic operations
        // on their ASCII values, avoiding a temporary variable.
        s[i] = s[i] + s[j];
        s[j] = s[i] - s[j];
        s[i] = s[i] - s[j];
      }
    }
  }

  /**
   * @brief Output generation block.
   * This loop iterates through the sorted string. It filters out the '+' characters
   * (which are now at the beginning) and prints the remaining digit characters,
   * separating them with '+' to reconstruct the sorted sum.
   */
  for (int i = 0; i < l; i++) {
    if (i == l - 1)
      // Pre-condition: This is the last character of the sorted string. It must be a digit.
      // Post-condition: The final digit is printed without a trailing '+'.
      cout << s[i];
    else if (s[i] != '+')
      // Pre-condition: The character is a digit, not a '+'.
      // Post-condition: The digit is printed, followed by a '+'.
      cout << s[i] << "+";
  }

  return 0;
}
