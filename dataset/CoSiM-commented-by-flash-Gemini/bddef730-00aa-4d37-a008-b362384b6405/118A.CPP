/**
 * @file 118A.CPP
 * @brief This program implements a string transformation task as part of a competitive programming problem.
 *   The task involves converting an input string by:
 *   1. Converting all uppercase letters to lowercase.
 *   2. Deleting all vowels ('a', 'o', 'y', 'e', 'u', 'i').
 *   3. Inserting a "." character before each consonant.
 *   Problem description: "String Task", Codeforces 118A.
 * * Algorithm: Single-pass string processing with filtering and transformation.
 *   It iterates through the input string, checking if each character is a vowel (case-insensitively).
 *   If it's a consonant, it prepends a '.' to the output and appends the lowercase version of the consonant.
 * Time Complexity: O(L) where L is the length of the input string, as it involves a single pass through the string.
 * Space Complexity: O(L) for storing the input string. The result is printed directly, not stored in a new string.
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries, such as <iostream>, <string>, <cctype>.
using namespace std;

/**
 * @brief Main function to perform the string transformation.
 * Functional Utility: Reads the input string, processes each character to filter vowels,
 *                     formats consonants by prepending a dot and converting to lowercase,
 *                     and prints the resulting modified string directly.
 * Note: `main()` without an explicit return type implies `int` in C++.
 */
main() {
  string s; // Stores the input string.
  // Block Logic: Reads the input string `s`.
  cin >> s;
  /**
   * Block Logic: Iterates through each character of the input string `s`.
   * Pre-condition: `s` is populated with the input string.
   * Invariant: Characters are processed for vowel/consonant filtering and formatted output.
   */
  for (int i = 0; i < s.length(); i++) {
    /**
     * Conditional Logic: Checks if the current character `s[i]` is a vowel (both lowercase and uppercase).
     * If it is a vowel, the `continue` statement skips to the next iteration, effectively deleting the vowel.
     */
    if (s[i] != 'a' && s[i] != 'A' && s[i] != 'e' && s[i] != 'E' &&
        s[i] != 'i' && s[i] != 'I' && s[i] != 'o' && s[i] != 'O' &&
        s[i] != 'U' && s[i] != 'u' && s[i] != 'y' && s[i] != 'y') {
      // If the character is a consonant (not a vowel).
      // Conditional Logic: Checks if the consonant is uppercase.
      if (isupper(s[i])) {
        // If uppercase, print a "." followed by its lowercase version.
        cout << "." << (char)tolower(s[i]);
      } else {
        // If lowercase, print a "." followed by the character as is.
        cout << "." << s[i];
      }
    }
  }
}
