/**
 * @file 118A.CPP
 * @brief This program implements a string transformation task as part of a competitive programming problem.
 *   The task involves converting an input string by:
 *   1. Converting all uppercase letters to lowercase.
 *   2. Deleting all vowels ('a', 'o', 'y', 'e', 'u', 'i').
 *   3. Inserting a "." character before each consonant.
 *   Problem description: "String Task", Codeforces 118A.
 * * Algorithm: String processing with character-by-character filtering and transformation.
 *   It iterates through the input string, applies rules to each character (manual lowercase conversion,
 *   vowel/consonant check), and builds a new string according to the rules.
 * Time Complexity: O(L) where L is the length of the input string, as it involves a single pass through the string.
 * Space Complexity: O(L) to store the processed characters in the `ret` string. In the worst case (all consonants), the output string length can be up to `2*L`.
 */
#include <iostream> // Required for standard input/output operations.
#include <string>   // Required for std::string.

using namespace std;

/**
 * @brief Main function to perform the string transformation.
 * Functional Utility: Reads the input string, applies the transformation rules (lowercase conversion,
 *                     vowel removal, consonant prepending with '.'), and prints the resulting string.
 */
int main() {
  string original, ret; // `original` stores the input string, `ret` builds the transformed string.

  // Block Logic: Reads the input string from standard input.
  cin >> original;

  /**
   * Block Logic: Iterates through each character of the `original` string to apply transformation rules.
   * Invariant: In each iteration, a character is processed: converted to lowercase if necessary,
   *            checked if it's a vowel, and formatted if it's a consonant to be added to `ret`.
   */
  int i;
  for (i = 0; i < original.length(); i++) {
    /**
     * Block Logic: Converts an uppercase character to lowercase using ASCII arithmetic.
     * Inline: `original[i] >= 65 && original[i] <= 90` checks if the character is in the ASCII range for uppercase English letters ('A' to 'Z').
     * Inline: `original[i] += 32` converts the uppercase character to lowercase by adding 32 (the ASCII difference between 'A' and 'a').
     */
    if (original[i] >= 65 && original[i] <= 90) {
      original[i] += 32;
    }
    /**
     * Block Logic: Checks if the current (now lowercase) character is a consonant.
     * Pre-condition: `original[i]` is in lowercase.
     * Inline: The condition `original[i] != 'a' && ... && original[i] != 'i'` explicitly defines
     *         which characters are considered vowels. If the character is NOT one of these vowels, it is a consonant.
     */
    if (original[i] != 'a' && original[i] != 'o' && original[i] != 'y' &&
        original[i] != 'e' && original[i] != 'u' && original[i] != 'i') {
      // If it's a consonant, prepend a '.' to the result string.
      ret.push_back('.');
      // Then append the consonant itself to the result string.
      ret.push_back(original[i]);
    }
  }

  // Block Logic: Prints the final transformed string followed by a newline.
  cout << ret << "\n";

  return 0;
}
