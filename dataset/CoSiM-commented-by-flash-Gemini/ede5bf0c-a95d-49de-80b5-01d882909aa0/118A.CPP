/**
 * @file 118A.CPP
 * @brief This program implements a string transformation task as part of a competitive programming problem.
 *   The task involves converting an input string by:
 *   1. Converting all uppercase letters to lowercase.
 *   2. Deleting all vowels ('a', 'o', 'y', 'e', 'u', 'i').
 *   3. Inserting a "." character before each consonant.
 *   Problem description: "String Task", Codeforces 118A.
 * * Algorithm: Two-pass string processing with filtering and transformation.
 *   1. A first pass converts all characters of the input string to lowercase in-place.
 *   2. A second pass iterates through the lowercase string, identifies consonants,
 *      and prints them prefixed with a ".". Vowels are skipped (deleted).
 * Time Complexity: O(L) where L is the length of the input string, as it involves two linear passes through the string.
 * Space Complexity: O(L) for storing the input string (modified in-place). The result is printed directly, not stored in a new string.
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries, such as <iostream>, <string>, <cctype>.
using namespace std;

/**
 * @brief Main function to perform the string transformation.
 * Functional Utility: Reads the input string, performs an in-place conversion to lowercase,
 *                     then filters out vowels and prints consonants prefixed with a dot.
 */
int main() {
  string s; // Stores the input string.
  // Block Logic: Reads the input string `s`.
  cin >> s;

  /**
   * Block Logic: First pass - Converts all characters in the string `s` to lowercase in-place.
   * Pre-condition: `s` is populated with the input string.
   * Invariant: After this loop, `s` will contain only lowercase characters.
   * Inline: `tolower(s[i])` converts the character at `s[i]` to its lowercase equivalent.
   */
  for (int i = 0; i < s.length(); i++) {
    s[i] = tolower(s[i]);
  }
  // cout << s; // This line is commented out, likely for debugging or testing.
  /**
   * Block Logic: Second pass - Iterates through the lowercase string to filter vowels and format consonants.
   * Pre-condition: `s` contains only lowercase characters.
   * Invariant: Consonants are printed prefixed with a dot; vowels are ignored.
   */
  for (int i = 0; i < s.length(); i++) {
    /**
     * Conditional Logic: Checks if the current character `s[i]` is a vowel (only lowercase check is needed now).
     * If it is a vowel, the `continue` statement skips to the next iteration, effectively deleting the vowel.
     */
    if (s[i] == 'a' || s[i] == 'e' || s[i] == 'i' || s[i] == 'o' ||
        s[i] == 'u' || s[i] == 'y')
      continue; // Skip vowels.
    else {
      // If the character is a consonant (not a vowel), print a '.' followed by the consonant.
      cout << "." << s[i];
    }
  }
  // Block Logic: Prints a newline character at the end of the output.
  cout << endl;

  return 0;
}
