/**
 * @file 71A.CPP
 * @brief This program solves the "Way Too Long Words" problem (Codeforces 71A).
 *   It abbreviates words longer than 10 characters to their first letter,
 *   the count of intermediate characters, and their last letter. Shorter words are printed as is.
 *   Problem description: https://codeforces.com/problemset/problem/71/A
 * * Algorithm: String length check and conditional formatting.
 *   For words longer than 10 characters, it calculates the number of characters
 *   between the first and last (which is `length - 2`) and formats the word.
 *   The calculation of the intermediate count uses an explicit loop, which is less
 *   efficient than direct arithmetic but yields the same result.
 * Time Complexity: O(N * L) where N is the number of words and L is the average length of a word.
 *   The inner loop to calculate `ile` contributes an additional factor of L for each long word.
 * Space Complexity: O(L_max) to store the longest input word.
 */
#include <iostream> // Required for standard input/output operations.
#define ll long long // Functional Utility: Alias for `long long`.
#define ld long double // Functional Utility: Alias for `long double` (not used in this code).
#include <string>   // Required for std::string.

/**
 * @brief Main function to process words and apply abbreviation rules.
 * Functional Utility: Reads the number of words, then processes each word
 *                     according to its length, applying abbreviation if necessary.
 */
int main() {
  int n; // Stores the total number of words to process.
  // Block Logic: Reads the total number of words `n`.
  std::cin >> n;
  /**
   * Block Logic: Iterates `n` times to process each word.
   * Pre-condition: `n` is a non-negative integer.
   * Invariant: Each iteration processes one word.
   */
  for (int i = 0; i < n; i++) {
    std::string word; // Stores the current word being processed.
    // Block Logic: Reads the current word.
    std::cin >> word;
    /**
     * Conditional Logic: Checks if the current word's length is greater than 10.
     * If true, the word is abbreviated. Otherwise, it's printed as is.
     */
    if (word.length() > 10) {
      int dlugosc = word.length() - 2; // `dlugosc`: Represents the number of characters between the first and last.
      std::cout << word[0];            // Prints the first character of the word.
      int ile = 0;                     // `ile`: Accumulates the count of intermediate characters.
      /**
       * Block Logic: Iterates `dlugosc` times to calculate `ile`.
       * Inline: This loop functionally calculates `ile = dlugosc`. A direct assignment
       *         `ile = dlugosc;` would be more efficient, but this approach also yields the correct count.
       */
      for (int j = 0; j < dlugosc; j++) {
        ile++;
      }
      // Prints the count of intermediate characters, then the last character.
      // Inline: `word[dlugosc + 1]` is equivalent to `word[word.length() - 1]` or `word.back()`.
      std::cout << ile << word[dlugosc + 1] << "\n";
    } else {
      std::cout << word << "\n"; // Prints the original word if its length is 10 or less.
    }
  }

  return 0; // Indicates successful program termination.
}