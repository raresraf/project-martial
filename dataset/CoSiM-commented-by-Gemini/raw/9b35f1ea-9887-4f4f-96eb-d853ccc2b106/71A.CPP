/**
 * @file 71A.CPP
 * @brief Abbreviates long words to a shorter format.
 *
 * This program implements a word abbreviation logic commonly seen in competitive
 * programming. It processes a series of words, and for each word, it checks
 * its length. If the word is longer than 10 characters, it is abbreviated by
 * taking its first character, followed by the count of characters between the
 * first and last character, and finally the last character. Words with 10 or
 * fewer characters are left unchanged.
 *
 * @example
 * "localization" becomes "l10n"
 * "internationalization" becomes "i18n"
 * "word" remains "word"
 *
 * @algorithm String length check and conditional formatting.
 * @time_complexity O(T * L) where T is the number of test cases and L is the
 * maximum length of a word. In practice, the string operations inside the loop are O(1)
 * because they only access first, last and length of the string. So the complexity is closer to O(T) if we don't consider the time to read the string.
 * @space_complexity O(L) to store the longest input word.
 */
#include <iostream>
#include <string>

using namespace std;

int main() {
  int t; // Represents the number of test cases to process.
  string str; // Will hold each word read from the input.
  
  // Read the total number of test cases.
  cin >> t;

  /**
   * @brief Main processing loop that iterates for each test case.
   * @invariant At the start of each iteration, `t` is decremented. The loop
   * continues until `t` becomes 0.
   */
  while (t--) {
    // Read the word for the current test case.
    cin >> str;

    /**
     * @brief Conditional block to check if the word needs abbreviation.
     * @pre `str` contains a word from the input.
     */
    if (str.length() > 10) {
      // If the word's length is greater than 10, abbreviate it.
      // The format is: first character, length-2, last character.
      cout << str[0] << str.length() - 2 << str[str.length() - 1] << endl;
    } else {
      // If the word is 10 characters or shorter, print it as is.
      cout << str << endl;
    }
  }

  return 0;
}