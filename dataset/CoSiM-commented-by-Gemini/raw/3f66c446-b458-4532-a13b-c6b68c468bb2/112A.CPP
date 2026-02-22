/**
 * @file 112A.CPP
 * @brief Solution for the "Petya and Strings" problem (Codeforces 112A).
 *
 * @algorithm
 * The problem requires a case-insensitive lexicographical comparison of two
 * strings. The output should be -1 if the first string is lexicographically
 * smaller, 1 if it is larger, and 0 if they are equal.
 *
 * This solution uses a standard and effective "convert-then-compare" strategy:
 * 1. Read the two input strings.
 * 2. Convert both strings entirely to lowercase. This makes the subsequent
 *    comparison case-insensitive.
 * 3. Use the built-in C++ string comparison operators (`>`, `==`) to determine
 *    the lexicographical order of the two lowercase strings.
 * 4. Print -1, 1, or 0 based on the result.
 *
 * @complexity
 * - Time Complexity: O(L), where L is the length of the strings, covering the
 *   time for reading, converting, and comparing.
 * - Space Complexity: O(L) to store the two input strings.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {

  string s1, s2;

  cin >> s1;
  cin >> s2;

  /**
   * @brief Convert both strings to lowercase to ensure a case-insensitive
   * comparison.
   */
  for (int i = 0; i < s1.length(); ++i) {

    s1[i] = tolower(s1[i]);
    s2[i] = tolower(s2[i]);
  }

  /**
   * @brief Perform the lexicographical comparison and print the result.
   * Note: Uses C-style `printf` for output.
   */
  if (s1 > s2)
    printf("1\n");
  else if (s1 == s2)
    printf("0\n");
  else // s1 < s2
    printf("-1\n");

  return 0;
}