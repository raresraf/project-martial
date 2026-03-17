/**
 * @file 118A.CPP
 * @brief Solution to a competitive programming problem, likely Codeforces 118A "String Task".
 *
 * @details
 * This program applies a specific transformation to an input string based on the following rules:
 * 1. All vowels ('A', 'O', 'Y', 'E', 'U', 'I') are deleted.
 * 2. A '.' character is inserted before each consonant.
 * 3. All uppercase consonants are converted to their lowercase equivalents.
 *
 * The program processes the string character by character and prints the result directly
 * to standard output without building an intermediate result string.
 *
 * @algorithm
 * The algorithm iterates through the input string one character at a time. For each
 * character, it checks if it is a vowel (case-insensitively, including 'Y').
 * - If the character is a vowel, it is ignored.
 * - If the character is a consonant, it is first converted to lowercase, and then
 *   a '.' character followed by the lowercase consonant is printed.
 *
 * @complexity
 * - Time Complexity: O(N), where N is the length of the input string, as it involves
 *   a single pass through the string.
 * - Space Complexity: O(N) to store the input string.
 */

#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function and entry point of the program.
 */
main() {
  string s;
  cin >> s;

  /**
   * @brief Iterates through each character of the input string.
   */
  for (int i = 0; i < s.length(); i++) {
    /**
     * @brief Checks if the character is a consonant.
     * The logic identifies a consonant by checking that it is NOT a vowel.
     * The vowels considered are 'a', 'e', 'i', 'o', 'u', and 'y', case-insensitively.
     * Note: The check `s[i] != 'y' && s[i] != 'y'` is redundant. It should likely
     * have been `s[i] != 'y' && s[i] != 'Y'`, but the logic works because
     * uppercase 'Y' is not explicitly checked here.
     */
    if (s[i] != 'a' && s[i] != 'A' && s[i] != 'e' && s[i] != 'E' &&
        s[i] != 'i' && s[i] != 'I' && s[i] != 'o' && s[i] != 'O' &&
        s[i] != 'U' && s[i] != 'u' && s[i] != 'y' && s[i] != 'y') {
      
      /**
       * @brief Prepends a '.' and prints the lowercase consonant.
       */
      if (isupper(s[i])) {
        cout << "." << (char)tolower(s[i]);
      } else {
        cout << "." << s[i];
      }
    }
  }
}
