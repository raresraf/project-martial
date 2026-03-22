/**
 * @file 118A.CPP
 * @brief Solution to Codeforces problem 118A - String Task.
 *        This program processes an input string by transforming it according to specific rules:
 *        1. All uppercase Latin letters are converted to their lowercase equivalents.
 *        2. All vowels ('a', 'o', 'y', 'e', 'u', 'i') are removed.
 *        3. A '.' character is inserted before each consonant.
 *
 * Algorithm:
 *   1. Reads an input string `s`.
 *   2. Initializes an empty string `r` to build the result.
 *   3. Iterates through each character `c` of the input string `s`.
 *      a. Converts `c` to lowercase if it's an uppercase letter.
 *      b. Checks if the modified `c` is a consonant (not 'a', 'e', 'i', 'o', 'u', 'y').
 *      c. If `c` is a consonant, appends '.' and then `c` to the result string `r`.
 *   4. Prints the final transformed string `r`.
 *
 * Time Complexity: O(L), where L is the length of the input string `s`, as it iterates through the string once.
 * Space Complexity: O(L) for storing the input string `s` and the result string `r`.
 */

#include <iostream>
#include <string>

using namespace std;

int main() {
  /**
   * @brief Main function that solves the "String Task" problem.
   *
   * Reads an input string, applies character-based transformations (lowercase conversion,
   * vowel removal, consonant prefixing with '.'), and prints the resulting string.
   *
   * @return int Returns 0 upon successful execution.
   */
  string s, r;
  cin >> s;
  /**
   * Block Logic: Iterates through each character of the input string to apply transformations.
   * Invariant: `c` represents the current character being processed from `s`. `r` accumulates
   *            the transformed string.
   * Pre-condition: `s` contains the input string.
   */
  for (char c : s) {
    /**
     * Conditional Logic: Converts uppercase letters to lowercase.
     * Pre-condition: `c` is the current character.
     */
    if (c >= 'A' && c <= 'Z') {
      c += 'a' - 'A'; // Converts to lowercase.
    }
    /**
     * Conditional Logic: Checks if the character is a consonant (not a, e, i, o, u, y).
     * If it is a consonant, appends '.' and then the consonant to the result string `r`.
     * Pre-condition: `c` is in lowercase.
     */
    if (c != 'a' && c != 'e' && c != 'i' && c != 'o' && c != 'u' && c != 'y') {
      r.append(1, '.'); // Appends a dot.
      r.append(1, c);   // Appends the consonant.
    }
  }
  // Prints the final transformed string.
  cout << r << endl;
  return 0;
}
