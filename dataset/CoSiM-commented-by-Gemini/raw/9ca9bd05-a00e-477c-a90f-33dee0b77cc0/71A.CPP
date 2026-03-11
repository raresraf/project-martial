/**
 * @file 71A.CPP
 * @brief Abbreviates long words to a shorter format.
 *
 * This program implements a word abbreviation logic commonly seen in competitive
 * programming. It processes a series of words, and for each word, it checks
 * its length. If the word is longer than 10 characters, it is abbreviated by
 * taking its first character, followed by the count of characters between the
 * first and last character, and finally the last character. This version explicitly uses `std::string`
 * concatenation with `std::to_string`.
 *
 * @example
 * "localization" becomes "l10n"
 * "internationalization" becomes "i18n"
 * "word" remains "word"
 *
 * @algorithm String length check and conditional formatting.
 * @time_complexity O(T * L), where T is the number of test cases and L is the
 * length of the word. The string concatenation might take O(L) time, making
 * the overall complexity dependent on the length of the abbreviated string.
 * However, since the output is short, it's practically very fast.
 * @space_complexity O(L) to store the longest input word.
 */
#include <cmath>
#include <iomanip>
#include <iostream>

int main() {
  int n;
  std::string word;

  std::cin >> n;
  // std::cout << n;

  /**
   * @brief Main processing loop that iterates for each test case.
   * @invariant The loop runs `n` times to process all words.
   */
  for (int i = 0; i < n; i++) {
    std::cin >> word;

    int wordLength = word.length();

    /**
     * @brief Conditional block to check if the word needs abbreviation.
     * @pre `word` contains a word from the input.
     */
    if (wordLength > 10) {
      // If the word's length is greater than 10, construct the abbreviated string.
      // This uses std::to_string for converting the length to a string.
      word =
          word.at(0) + std::to_string(wordLength - 2) + word.at(wordLength - 1);
    }

    std::cout << word << std::endl;
  }

  return 0;
}
