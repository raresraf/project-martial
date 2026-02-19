/**
 * @file 112A.CPP
 * @brief Solves the "Petya and Strings" problem by performing a case-insensitive string comparison.
 * @details This program compares two strings lexicographically, ignoring differences in
 * character case. It outputs -1 if the first string is lexicographically smaller, 1 if it
 * is larger, and 0 if they are equal.
 *
 * @algorithm The algorithm iterates through the two strings simultaneously, character by
 * character. At each position, it converts the characters from both strings to
 * uppercase to achieve case-insensitivity. It then compares these characters.
 * - If the characters differ, the lexicographical order is determined, the result
 *   is set (1 or -1), and the loop terminates.
 * - If the characters are the same, the loop continues.
 * - If the loop finishes without finding any differences, the strings are equal,
 *   and the result is 0.
 *
 * @time_complexity O(L), where L is the length of the strings, as it involves a single
 * pass through the characters.
 * @space_complexity O(L), to store the two input strings.
 */
#include <ctype.h>
#include <iostream>
#include <string>
using namespace std;
int main() {
  string first, second;
  cin >> first;
  cin >> second;
  // flag: Stores the result of the comparison. 0 for equal, 1 for greater, -1 for less.
  int flag = 0;
  /**
   * Block Logic: Iterate through the strings to find the first point of difference,
   * comparing characters case-insensitively.
   */
  for (int i = 0; i < first.length(); i++) {
    // The `toupper` function ensures that the comparison is case-insensitive.
    if (toupper(first[i]) == toupper(second[i])) {
      // Characters are equal (case-insensitively), so continue to the next pair.
      continue;
    } else if (toupper(first[i]) > toupper(second[i])) {
      // The first string is lexicographically greater.
      flag++;
      break;
    } else {
      // The second string is lexicographically greater.
      flag--;
      break;
    }
  }

  cout << flag;

  return 0;
}