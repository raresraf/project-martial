/* Sometimes some words like "localization" or "internationalization" are so
long that writing them many times in one text is quite tiresome.

Let's consider a word too long, if its length is strictly more than 10
characters. All too long words should be replaced with a special abbreviation.

This abbreviation is made like this: we write down the first and the last letter
of a word and between them we write the number of letters between the first and
the last letters. That number is in decimal system and doesn't contain any
leading zeroes.

Thus, "localization" will be spelt as "l10n", and "internationalization  will be
spelt as "i18n".

You are suggested to automatize the process of changing the words with
abbreviations. At that all too long words should be replaced by the abbreviation
and the words that are not too long should not undergo any changes.

Input
The first line contains an integer n (1   n   100). Each of the following n
lines contains one word. All the words consist of lowercase Latin letters and
possess the lengths of from 1 to 100 characters.

Output
Print n lines. The i-th line should contain the result of replacing of the i-th
word from the input data. */

/**
 * @file 71A.CPP
 * @brief Solution to the "Way Too Long Words" problem from Codeforces.
 *
 * This program abbreviates words that are strictly longer than 10 characters.
 * The abbreviation is formed by taking the first letter, the count of
 * letters between the first and last, and the last letter. Shorter words
 * remain unchanged.
 *
 * @algorithm String Manipulation
 * @time_complexity O(N*L), where N is the number of words and L is the max length of a word. In this problem, it's effectively O(N) because L is small and constant.
 * @space_complexity O(L) to store the longest word.
 */
#include <iostream>
#include <string>

using namespace std;

int main() {
  int t;
  string word;
  cin >> t;
  /**
   * @brief Processes each word from the input.
   */
  while (t--) {
    cin >> word;
    // Check if the word's length is greater than 10.
    if (word.length() > 10) {
      /**
       * @brief Constructs and prints the abbreviation for a long word.
       * The abbreviation consists of the first character, the number of
       * internal characters, and the last character.
       */
      string new_word;
      string first(1, word[0]);
      string last(1, word[word.length() - 1]);
      new_word = first + to_string(word.length() - 2) + last;
      cout << new_word << endl;
    } else
      // If the word is not too long, print it as is.
      cout << word << endl;
  }
  return 0;
}
