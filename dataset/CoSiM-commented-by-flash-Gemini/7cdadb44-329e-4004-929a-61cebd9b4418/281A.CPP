/**
 * @file 281A.CPP
 * @brief Solution to Codeforces problem 281A - Word Capitalization.
 *        This program takes a single word as input and capitalizes its first letter,
 *        leaving the rest of the letters unchanged.
 *
 * Algorithm:
 *   1. Initializes a boolean flag `done` to `false`, which indicates whether the first
 *      character has been processed and potentially capitalized.
 *   2. Reads characters from standard input one by one until the end of the input.
 *   3. For each character `c`:
 *      a. If `done` is `false` (meaning it's the very first character of the word):
 *         i. Checks if `c` is a lowercase letter.
 *         ii. If it is, converts `c` to its uppercase equivalent.
 *         iii. Sets `done` to `true` to ensure no subsequent characters are capitalized.
 *      b. Prints the character `c` (either original or capitalized).
 *   4. After processing all characters, prints a newline.
 *
 * Time Complexity: O(L), where L is the length of the input word, as it processes each character once.
 * Space Complexity: O(1) for storing a few character and boolean variables.
 */

#include <iostream>

int main() {
  /**
   * @brief Main function that solves the "Word Capitalization" problem.
   *
   * Reads a word character by character, capitalizes only its first letter if it's lowercase,
   * and then prints the modified word.
   *
   * @return int Returns 0 upon successful execution.
   */
  char c;       // Variable to store the current character being read.
  bool done = false; // Flag to track if the first character has been processed.
  /**
   * Block Logic: Reads characters from standard input one by one until EOF.
   * Invariant: `c` holds the current character. `done` is true after the first character.
   * Pre-condition: Input is a word potentially containing lowercase and uppercase letters.
   */
  while (std::cin >> c) {
    /**
     * Conditional Logic: Applies capitalization only to the very first character.
     * Pre-condition: `done` is false, indicating this is the first character.
     */
    if (!done) {
      /**
       * Conditional Logic: Checks if the first character is a lowercase letter.
       * If it is, converts it to uppercase.
       */
      if ((c >= 'a') && (c <= 'z'))
        c += 'A' - 'a'; // Transforms lowercase to uppercase.
      done = true;      // Marks that the first character has been handled.
    }
    std::cout << c; // Prints the character (either modified or original).
  }
  std::cout << '\n'; // Prints a newline at the end.
  return 0;
}
