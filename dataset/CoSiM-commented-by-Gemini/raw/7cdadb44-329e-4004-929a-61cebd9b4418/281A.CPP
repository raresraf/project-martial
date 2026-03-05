/**
 * @file 281A.CPP
 * @brief Solution for Codeforces Problem 281A, "Word Capitalization".
 * @details This program capitalizes the first letter of a single input word.
 *
 * Algorithm:
 * The code uses a character-by-character streaming approach.
 * 1. It reads the input one character at a time. The `>>` operator skips any
 *    leading whitespace.
 * 2. A boolean flag `done` is used to track whether the first character has
 *    already been processed.
 * 3. For the first character read (`!done`), it checks if it is a lowercase
 *    letter and, if so, converts it to its uppercase equivalent.
 * 4. The `done` flag is then set to `true`.
 * 5. The (potentially modified) character is printed.
 * 6. All subsequent characters are read and printed without modification as the
 *    `!done` block is no longer entered.
 *
 * Time Complexity: O(L), where L is the length of the word.
 * Space Complexity: O(1).
 */
#include <iostream>

int main() {
  char c;
  bool done = false;
  /**
   * @brief Reads the input stream character by character. As `>> c` skips
   * whitespace, this effectively reads all non-whitespace characters
   * consecutively.
   */
  while (std::cin >> c) {
    /**
     * @brief This block executes only once for the first character of the word.
     */
    if (!done) {
      // If the character is lowercase, convert it to uppercase.
      if ((c >= 'a') && (c <= 'z'))
        c += 'A' - 'a';
      done = true; // Ensure this block doesn't run again.
    }
    std::cout << c;
  }
  std::cout << '\n';
}
