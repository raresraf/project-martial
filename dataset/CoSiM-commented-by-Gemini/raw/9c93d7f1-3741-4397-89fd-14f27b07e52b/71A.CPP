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
 * maximum length of a word. The string operations inside the loop are O(1) in practice.
 * @space_complexity O(L) to store the longest input word.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  // The line below is commented out, but it is used for redirecting file input,
  // a common practice in competitive programming for local testing.
  // freopen("input.txt", "r", stdin);
  int t; // Represents the number of test cases to process.
  string s; // Will hold each word read from the input.

  /**
   * @brief A compact for-loop to handle test cases. It initializes `t` with
   * the number of test cases, and the loop continues as long as `t` is non-zero,
   * decrementing `t` in each iteration.
   */
  for (scanf("%d", &t); t--;) {
    // Read the word for the current test case.
    cin >> s;
    
    /**
     * @brief Conditional block to check if the word needs abbreviation.
     * @pre `s` contains a word from the input.
     */
    if (s.size() <= 10) {
      // If the word is 10 characters or shorter, print it as is.
      cout << s << endl;
    } else {
      // If the word's length is greater than 10, abbreviate it.
      // The format is: first character, length-2, last character.
      cout << s[0] << (s.size() - 2) << s[s.size() - 1] << endl;
    }
  }
  return 0;
}
