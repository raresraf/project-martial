/**
 * @file 112A.CPP
 * @brief Solution to the "Petya and Strings" problem (Codeforces 112A).
 * @details The problem requires a case-insensitive lexicographical comparison of two strings.
 *          The output should be -1 if the first string is lexicographically smaller,
 *          1 if it is larger, and 0 if they are equal.
 *
 * @algorithm The algorithm first standardizes the strings by converting them both to the
 *            same case (lowercase in this implementation). After conversion, it performs
 *            a direct lexicographical comparison on the modified strings. C++'s standard
 *            string comparison operators naturally provide the desired -1, 0, or 1
 *            logic when applied in an if-else if-else structure.
 *
 * @time_complexity O(L), where L is the length of the strings, due to the need to
 *                  iterate through them for case conversion and comparison.
 * @space_complexity O(L) to store the two strings.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  string s1, s2;
  // This loop structure will handle one or more pairs of string inputs.
  while (cin >> s1 >> s2) {
    /**
     * @brief Convert both strings to lowercase for case-insensitive comparison.
     */
    for (int i = 0; i < s1.size(); i++) {
      s1[i] = tolower(s1[i]);
      s2[i] = tolower(s2[i]);
    }

    /**
     * @brief Perform lexicographical comparison on the lowercase strings.
     * The standard C++ string operators (<, ==, >) are used.
     */
    if (s1 == s2)
      cout << "0" << endl;
    else if (s1 < s2)
      cout << "-1" << endl;
    else
      cout << "1" << endl;
  }
  return 0;
}
