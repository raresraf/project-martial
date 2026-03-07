/**
 * @file 118A.CPP
 * @brief Solution to a competitive programming problem that transforms an input string.
 * The transformation involves removing all vowels and inserting a '.' character
 * before each remaining consonant, converting the consonant to lowercase.
 * Algorithm: String iteration and conditional character processing.
 * Time Complexity: O(L), where L is the length of the input string.
 * Space Complexity: O(L) for the resulting transformed string.
 */
#include "string"   // Required for string manipulation.
#include <iostream> // Required for input/output operations.
// The `towlower` function, if not implicitly included by iostream or string,
// would typically require <cctype> or <wctype.h> for its declaration.
// In competitive programming, <bits/stdc++.h> often includes it.
using namespace std;

int main(int argc, const char *argv[]) {
  // Declares 'str' to store the input string and 'res' to build the result string.
  string str, res;
  // Reads the input string from standard input.
  cin >> str;

  /**
   * Block Logic: Iterates through each character of the input string.
   * Invariant: At the start of each iteration, 'i' is the current character's index.
   * Post-condition: 'res' contains the transformed string.
   */
  for (int i = 0; i < str.length(); i++) {
    /**
     * Conditional Logic: Checks if the current character is a vowel (case-insensitive).
     * If it's a vowel, it's skipped.
     */
    if (str[i] == 'a' || str[i] == 'e' || str[i] == 'i' || str[i] == 'o' ||
        str[i] == 'u' || str[i] == 'y' || str[i] == 'A' || str[i] == 'E' ||
        str[i] == 'O' || str[i] == 'I' || str[i] == 'U' || str[i] == 'Y')
      continue; // Skips to the next character if it's a vowel.
    else {
      // If it's a consonant, a '.' is appended to the result string.
      res += '.';
      // The consonant is then converted to lowercase using `towlower` and appended to the result string.
      // `towlower` correctly handles locale-specific character casing.
      res += towlower(str[i]);
    }
  }
  // Prints the final transformed string.
  cout << res;

  return 0; // Indicates successful program execution.
}
