/**
 * @file 112A.CPP
 * @brief Performs a case-insensitive lexicographical comparison of two strings.
 *
 * This program reads two strings of equal length and compares them character by
 * character, ignoring case. It outputs -1 if the first string is lexicographically
 * smaller, 1 if it is larger, and 0 if they are equal.
 *
 * @algorithm Lexicographical Comparison
 * @complexity
 * - Time: O(L), where L is the length of the strings, because it may iterate
 *   through the entire string in the worst case (if the strings are equal).
 * - Space: O(L) to store the two input strings.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  string a, b;
  cin >> a >> b;
  int len = a.length();

  /**
   * @brief Block Logic: Iterates through the strings to find the first point of difference.
   * Invariant: The prefixes of the strings of length `i` are lexicographically
   *            equal (case-insensitively).
   */
  for (int i = 0; i < len; i++) {
    /**
     * @brief Pre-condition: Compares the lowercase versions of the characters at the current position.
     * The explicit cast to (int) is not strictly necessary as chars are promoted,
     * but it makes the integer comparison explicit.
     */
    if ((int)tolower(a[i]) > (int)tolower(b[i])) {
      cout << 1;
      return 0;
    }
    if ((int)tolower(a[i]) < (int)tolower(b[i])) {
      cout << -1;
      return 0;
    }
  }

  // If the loop completes, the strings are lexicographically equal (case-insensitively).
  cout << 0;
}