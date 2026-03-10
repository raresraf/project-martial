/**
 * @file 118A.CPP
 * @brief This program implements a string transformation task as part of a competitive programming problem.
 *   The task involves converting an input string by:
 *   1. Converting all uppercase letters to lowercase.
 *   2. Deleting all vowels ('a', 'o', 'y', 'e', 'u', 'i').
 *   3. Inserting a "." character before each consonant.
 *   Problem description: "String Task", Codeforces 118A.
 * * Algorithm: String processing with character-by-character filtering and transformation.
 *   It iterates through the input string, checking if each character is a vowel (case-insensitively).
 *   If it's a consonant, it prepends a '.' to a new string and appends the lowercase version of the consonant.
 * Time Complexity: O(L) where L is the length of the input string, as it involves a single pass through the string.
 * Space Complexity: O(L) to store the processed characters in the `s2` string. In the worst case (all consonants), the output string length can be up to `2*L`.
 */
#include <iostream> // Required for standard input/output operations.
#include <string>   // Required for std::string.
#include <cctype>   // Required for std::tolower.

using namespace std;

/**
 * @brief Main function to perform the string transformation.
 * Functional Utility: Reads the input string, applies the transformation rules (vowel removal,
 *                     consonant prepending with '.', and lowercase conversion), and
 *                     builds the resulting string `s2`.
 */
int main() {
  string s, s2; // `s` stores the original input string, `s2` builds the transformed string.
  // Block Logic: Reads the input string `s`.
  cin >> s;
  int l = s.length(); // Stores the length of the input string.

  /**
   * Block Logic: Iterates through each character of the input string `s`.
   * Pre-condition: `s` is populated with the input string.
   * Invariant: `s2` is progressively built with transformed characters.
   */
  for (int i = 0; i < l; i++) {
    /**
     * Conditional Logic: Checks if the current character `s[i]` is a vowel (both lowercase and uppercase).
     * If it is a vowel, the `continue` statement skips to the next iteration, effectively deleting the vowel.
     */
    if (s[i] == 'a' || s[i] == 'e' || s[i] == 'i' || s[i] == 'o' ||
        s[i] == 'u' || s[i] == 'y' || s[i] == 'A' || s[i] == 'E' ||
        s[i] == 'I' || s[i] == 'O' || s[i] == 'U' || s[i] == 'Y') {
      continue; // Skip vowels.
    }
    // If the character is a consonant (not a vowel), prepend a '.' to `s2`.
    s2 += ".";
    // Then append the lowercase version of the consonant to `s2`.
    // Inline: `tolower(s[i])` converts the character to its lowercase equivalent.
    s2 += tolower(s[i]);
  }
  return 0;
}