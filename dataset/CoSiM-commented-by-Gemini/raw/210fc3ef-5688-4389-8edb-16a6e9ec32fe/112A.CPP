/**
 * @file 112A.CPP
 * @brief Solution for Codeforces Problem 112A, "Petya and Strings".
 * @details This program performs a case-insensitive lexicographical comparison of two input strings. It outputs -1 if the first string is lexicographically smaller than the second, 1 if it is larger, and 0 if they are equal.
 *
 * @algorithm Iterative case-insensitive character comparison.
 * @description The program iterates through both strings simultaneously. In each iteration, it converts the characters at the current position to lowercase and compares them. If a difference is found, it sets the result and breaks the loop. If the loop completes without finding any differences, the strings are equal, and the result remains 0.
 * @complexity
 * - Time: O(L), where L is the length of the strings, as it performs a single pass.
 * - Space: O(1), excluding the storage for the input strings themselves.
 */

#include <iostream>

using namespace std;

int main() {
  string a, b;
  cin >> a >> b;
  char c1, c2;
  int result = 0;

  /**
   * @brief Iterates through the strings to find the first point of difference.
   * @invariant `result` is 0, indicating that the prefixes of the strings examined so far are equal.
   */
  for (int i = 0; i < a.length(); i++) {
    // Convert characters to lowercase for case-insensitive comparison.
    c1 = tolower(a[i]);
    c2 = tolower(b[i]);

    /**
     * @brief Lexicographically compares the lowercase characters.
     * If a difference is found, the result is set, and the loop is terminated
     * as the overall result is now determined.
     */
    if (c1 < c2) {
      result = -1;
      break;
    } else if (c1 > c2) {
      result = 1;
      break;
    }
  }

  cout << result << "\n";
}