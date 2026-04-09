/**
 * @file 118A.CPP
 * @brief This program implements a string transformation logic.
 * It reads a single word, filters out all vowels (case-insensitive),
 * and for each remaining consonant, converts it to lowercase and
 * prepends it with a ".". The transformed string is then printed.
 *
 * Algorithm: String traversal, character classification, case conversion.
 * It iterates through the input string, character by character.
 * For each character, it checks if it's a vowel (A, E, I, O, U, Y, case-insensitive).
 * If it's a consonant, it converts it to lowercase and prints a "." followed by the character.
 *
 * Time Complexity: O(N), where N is the length of the input string.
 * Each character of the input string is processed exactly once.
 *
 * Space Complexity: O(1) additional space, as the output is streamed directly
 * and no significant intermediate data structures are stored.
 */

#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function of the program.
 * Orchestrates the string transformation process.
 * Reads an input string, applies the vowel filtering and consonant formatting rules,
 * and prints the resulting transformed string to standard output.
 * @pre Standard input contains a single string `s`.
 * @post The transformed string, with vowels removed and consonants lowercased and prepended by '.', is printed to standard output.
 * @return 0 upon successful execution, indicating the program finished without errors.
 */
int main() {
  int i;
  string s;
  // Functional Utility: Reads the input string from standard input.
  cin >> s;
  /**
   * Block Logic: Iterates through each character of the input string `s`.
   * Invariant: At the start of each iteration, characters `s[0...i-1]` have been processed.
   * `i` serves as the current character index.
   */
  for (i = 0; i < s.size(); i++) {
    /**
     * Block Logic: Checks if the current character `s[i]` is a vowel (case-insensitive).
     * If `s[i]` is a vowel, it is skipped (not printed).
     * Pre-condition: `s[i]` contains the character to be classified.
     */
    if ((s[i] != 'A') && (s[i] != 'E') && (s[i] != 'I') && (s[i] != 'O') &&
        (s[i] != 'U') && (s[i] != 'Y') && (s[i] != 'a') && (s[i] != 'e') &&
        (s[i] != 'i') && (s[i] != 'o') && (s[i] != 'u') && (s[i] != 'y')) {
      // Functional Utility: Prints a '.' before each consonant.
      cout << ".";
      /**
       * Block Logic: Checks if the consonant is uppercase and converts it to lowercase if necessary, then prints it.
       * If `s[i]` is already lowercase, it is printed as is.
       * Pre-condition: `s[i]` is a consonant.
       */
      if (s[i] >= 'A' && s[i] <= 'Z') {
        // Inline: Converts the uppercase character `s[i]` to its lowercase equivalent.
        s[i] = tolower(s[i]);
        // Functional Utility: Prints the lowercase consonant.
        cout << s[i];
      } else
        // Functional Utility: Prints the consonant as is (already lowercase).
        cout << s[i];
    }
  }
  return 0;
}
