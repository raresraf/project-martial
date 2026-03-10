/**
 * @file 118A.CPP
 * @brief This program implements a string transformation task as part of a competitive programming problem.
 *   The task involves converting an input string by:
 *   1. Converting all uppercase letters to lowercase.
 *   2. Deleting all vowels ('a', 'o', 'y', 'e', 'u', 'i').
 *   3. Inserting a "." character before each consonant.
 *   Problem description: "String Task", Codeforces 118A.
 * * Algorithm: Single-pass string processing with `switch` statement filtering.
 *   It iterates through the input string. For each character, a `switch` statement
 *   efficiently checks if it's a vowel (case-insensitively). If it's a consonant,
 *   it prints "." followed by the lowercase version of the character.
 * Time Complexity: O(L) where L is the length of the input string, as it involves a single pass through the string.
 * Space Complexity: O(L) for storing the input string. The result is printed directly, not stored in a new string.
 */
/*
~ Author    : @tridib_2003
~ Title     : 118A. String Task
~ Link      : https://codeforces.com/problemset/problem/118/A
*/

#include <bits/stdc++.h> // Includes most standard C++ libraries, such as <iostream>, <string>, <cctype>.

using namespace std;

/**
 * @brief Main function to perform the string transformation.
 * Functional Utility: Configures I/O, reads the input string, filters vowels,
 *                     formats consonants, and prints the resulting modified string.
 */
int main() {

  // Functional Utility: Optimizes C++ standard I/O streams for faster execution by
  //                     disabling synchronization with C stdio and untying cin from cout.
  ios::sync_with_stdio(false);
  cin.tie(0);
  /*
      Block Logic: Commented-out section, likely a common competitive programming template
      structure for handling multiple test cases. Not active in this solution.
      int tc;
  cin>>tc;
  while(tc--) {
      }
      */

  string str; // Stores the input string.
  // Block Logic: Reads the input string `str`.
  cin >> str;

  /**
   * Block Logic: Iterates through each character of the input string to apply transformation rules.
   * A `switch` statement is used for efficient character classification.
   * Pre-condition: `str` is populated with the input string.
   * Invariant: For each character, if it's a vowel, it's skipped; otherwise, it's printed formatted.
   */
  for (int i = 0; i < str.length(); i++) {
    switch (str[i]) {
    // Case Labels: Matches lowercase vowels. If matched, `continue` skips to the next character.
    case 'a':
    case 'e':
    case 'i':
    case 'o':
    case 'u':
    case 'y':
    // Case Labels: Matches uppercase vowels. The expression `'a' - ' '` is a common trick
    // to derive the ASCII value of 'A' (or other uppercase letters) from their lowercase counterparts.
    // For example, 'a' - ' ' (ASCII 97 - 32) = 65, which is 'A'.
    case 'a' - ' ': // 'A'
    case 'e' - ' ': // 'E'
    case 'i' - ' ': // 'I'
    case 'o' - ' ': // 'O'
    case 'u' - ' ': // 'U'
    case 'y' - ' ': // 'Y'
      continue;     // Skips processing for vowels (effectively deleting them).
    default:        // If the character is not a vowel (i.e., it's a consonant).
      // Prints a "." followed by the lowercase version of the consonant.
      // Inline: `(char)tolower(str[i])` converts the character to lowercase and casts back to char.
      cout << "." << (char)tolower(str[i]);
    }
  }
  return 0; // Indicates successful program termination.
}
