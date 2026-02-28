/**
 * @file 71A.CPP
 * @brief Solution for the "Way Too Long Words" problem from Codeforces.
 * @details This program processes a list of words. If a word is longer than 10
 * characters, it is abbreviated. Otherwise, it is printed as is. The abbreviation
 * consists of the first character, the number of characters between the first and
 * last, and the last character. For example, "localization" becomes "l10n".
 *
 * @algorithm String Length Check and Manipulation
 * 1. Read the number of words, `n`.
 * 2. Loop `n` times.
 * 3. In each iteration, read a word.
 * 4. Check if the word's length is greater than 10.
 * 5. If it is, construct the abbreviation:
 *    a. Take the first character.
 *    b. Calculate the number of middle characters (length - 2).
 *    c. Take the last character.
 *    d. Print the first character, the count, and the last character.
 * 6. If the length is 10 or less, print the word unchanged.
 *
 * @complexity
 * - Time: O(N * L_max), where N is the number of words and L_max is the maximum
 *   length of a word. This is dominated by the time to read the input strings.
 * - Space: O(L_max), to store the longest word.
 */

#include <iostream>
#include <string>

// Unused macros, but common in competitive programming templates.
#define ll long long
#define ld long double

int main() {
  // n: the number of words to process.
  int n;
  std::cin >> n;

  /**
   * @brief Loop to process each of the `n` words.
   */
  for (int i = 0; i < n; i++) {
    std::string word;
    std::cin >> word;

    // Pre-condition: Check if the word is "too long".
    if (word.length() > 10) {
      // `dlugosc` (Polish for "length") stores the count of characters
      // between the first and last characters.
      int dlugosc = word.length() - 2;
      
      // Print the first character of the word.
      std::cout << word[0];

      // `ile` (Polish for "how many") is used to store the same value as `dlugosc`.
      int ile = 0;
      
      /**
       * @brief Redundant loop to calculate the length.
       * This loop is unnecessary. The value of `ile` after the loop is simply
       * `dlugosc`. Printing `dlugosc` directly would be more efficient.
       */
      for (int j = 0; j < dlugosc; j++) {
        ile++;
      }
      
      // Print the calculated number of middle characters.
      std::cout << ile;
      
      // Print the last character of the word.
      // The index is `dlugosc + 1`, which equals `(word.length() - 2) + 1`,
      // correctly evaluating to `word.length() - 1`.
      std::cout << word[dlugosc + 1] << "\n";
    } else {
      // If the word is not too long, print it as is.
      std::cout << word << "\n";
    }
  }

  return 0; // Indicate successful execution.
}
