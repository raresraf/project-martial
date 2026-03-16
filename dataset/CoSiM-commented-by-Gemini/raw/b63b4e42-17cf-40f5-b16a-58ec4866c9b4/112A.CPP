/**
 * @file 112A.CPP
 * @brief Solution to the "Petya and Strings" problem (Codeforces 112A).
 *
 * @details
 * This program compares two strings lexicographically, ignoring the case of the letters.
 * It reads two strings of equal length and determines their order.
 *
 * The output is determined as follows:
 * - -1: if the first string is lexicographically smaller than the second.
 * -  1: if the first string is lexicographically larger than the second.
 * -  0: if the strings are equal.
 *
 * The comparison is done character by character. At the first position where the
 * strings differ, their lexicographical order is determined. To ignore case,
 * each character is converted to lowercase before comparison.
 *
 * Algorithm:
 * 1. Read two strings.
 * 2. Iterate from the first to the last character of the strings.
 * 3. In each iteration, compare the lowercase versions of the characters at the
 *    current position.
 * 4. If a difference is found, set the result (-1 or 1) and stop.
 * 5. If the loop finishes without finding any differences, the strings are equal,
 *    and the result is 0.
 *
 * Time Complexity: O(L), where L is the length of the strings. The loop runs at most L times.
 * Space Complexity: O(L), to store the two input strings.
 */
#include <iostream>
#include <string> // Required for std::string
#include <cctype>  // Required for std::tolower

using namespace std;

int main() {
  string first;
  getline(cin, first);
  string second;
  getline(cin, second);
  int comparison = 0; // Default result is 0 (equal)

  /**
   * @brief Performs a case-insensitive, character-by-character comparison.
   * @details The loop iterates through both strings simultaneously. It converts
   * characters to lowercase to ensure the comparison is case-insensitive.
   * As soon as a lexicographical difference is found, the result is set,
   * and the loop terminates.
   */
  for (int k = 0; k < first.size(); ++k) {
    if (tolower(first[k]) < tolower(second[k])) {
      comparison = -1;
      break;
    }
    if (tolower(first[k]) > tolower(second[k])) {
      comparison = 1;
      break;
    }
  }

  // Using printf from <cstdio> which is included via <iostream> in some implementations.
  // std::cout << comparison << std::endl; would be more idiomatic C++.
  printf("%d\n", comparison);
  return 0;
}