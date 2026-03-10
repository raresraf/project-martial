/**
 * @file 112A.CPP
 * @brief This program solves the "Petya and Strings" problem (Codeforces 112A).
 *   It compares two input strings lexicographically, ignoring case.
 *   Problem description: https://codeforces.com/problemset/problem/112/A
 * * Algorithm: Case-insensitive lexicographical comparison.
 *   1. Reads two input strings.
 *   2. Converts all characters in both strings to lowercase in-place.
 *   3. Compares the resulting lowercase strings using `std::string::compare`.
 *   4. Prints the result of the comparison (0 if equal, negative if first is smaller, positive if first is larger).
 * Time Complexity: O(L) where L is the length of the strings, due to iterating through strings
 *   for lowercase conversion and comparison.
 * Space Complexity: O(L) to store the input strings (modified in-place).
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries, such as <iostream>, <string>, <algorithm>.
using namespace std;

/**
 * @brief Main function to perform case-insensitive string comparison.
 * Functional Utility: Reads two strings, converts them to lowercase,
 *                     compares them lexicographically using `std::string::compare`,
 *                     and prints the comparison result.
 */
int main() {

  string s1, s2; // `s1` and `s2` store the two input strings.
  // `c1`, `c2`: Declared but unused variables, likely remnants from a template or previous attempts.
  int c1 = 0, c2 = 0;
  // Block Logic: Reads the two input strings into `s1` and `s2`.
  cin >> s1 >> s2;
  /**
   * Block Logic: Converts `s1` to lowercase in-place using a range-based for loop.
   * Pre-condition: `s1` is populated with the input string.
   * Invariant: After this loop, `s1` will contain only lowercase characters.
   * Inline: `if (i < 'a') i += 32;` checks if the character `i` is uppercase (ASCII value less than 'a')
   *         and converts it to lowercase by adding 32 (ASCII difference).
   */
  for (char &i : s1) { // Uses a reference `&i` to modify the character in-place.
    if (i < 'a')
      i += 32;
  }
  /**
   * Block Logic: Converts `s2` to lowercase in-place using a range-based for loop.
   * Pre-condition: `s2` is populated with the input string.
   * Invariant: After this loop, `s2` will contain only lowercase characters.
   */
  for (char &i : s2) {
    if (i < 'a')
      i += 32;
  }
  // Block Logic: Compares the two lowercase strings lexicographically using `std::string::compare`.
  // `compare` returns 0 if strings are equal, a negative value if `s1` is smaller, and a positive value if `s1` is larger.
  cout << (s1.compare(s2)) << endl;

  return 0; // Indicates successful program termination.
}
