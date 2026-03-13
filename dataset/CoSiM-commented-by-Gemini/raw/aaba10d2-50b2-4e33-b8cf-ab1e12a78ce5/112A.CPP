/**
 * @file 112A.CPP
 * @brief Performs a case-insensitive lexicographical comparison of two strings.
 *
 * This program reads two strings, converts them to lowercase, and then compares them.
 * It outputs -1 if the first string is lexicographically smaller than the second,
 * 1 if it's larger, and 0 if they are equal.
 *
 * Algorithm:
 * 1. Read two strings, s1 and s2.
 * 2. Iterate through both strings to convert all characters to lowercase.
 * 3. Perform a direct lexicographical comparison on the modified, lowercase strings.
 * 4. Print -1, 1, or 0 based on the comparison result.
 *
 * Time Complexity: O(L), where L is the length of the strings. This is dominated
 * by the character conversion and the final comparison.
 * Space Complexity: O(L), to store the two input strings.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  string s1, s2;
  int i, x; // 'i' is used as a loop counter, 'x' is declared but unused.

  // The while loop will continue as long as pairs of strings can be read from input.
  while (cin >> s1 >> s2) {
    /**
     * Block Logic: Convert both strings to lowercase for case-insensitive comparison.
     * Invariant: After this loop, s1 and s2 will contain only lowercase characters.
     */
    for (i = 0; i < s1.size(); i++) {
      s1[i] = tolower(s1[i]);
      s2[i] = tolower(s2[i]);
    }
    /**
     * Block Logic: Compare the two lowercase strings and print the result.
     * C++ string comparison operators perform lexicographical comparison.
     */
    if (s1 == s2)
      cout << "0" << endl; // Strings are equal.
    if (s1 < s2)
      cout << "-1" << endl; // s1 comes before s2 alphabetically.
    if (s1 > s2)
      cout << "1" << endl; // s1 comes after s2 alphabetically.
  }
  return 0;
}
