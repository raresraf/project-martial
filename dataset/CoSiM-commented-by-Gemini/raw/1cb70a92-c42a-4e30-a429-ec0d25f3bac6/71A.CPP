/**
 * @file 71A.CPP
 * @brief Solution for Codeforces problem 71A, "Way Too Long Words".
 *
 * @description
 * This program processes a list of words and abbreviates any word that is
 * longer than 10 characters. The abbreviation is formed by taking the first
 * character, followed by the count of characters in between, and ending with
 * the last character. Words with 10 or fewer characters are left unchanged.
 *
 * @example
 * "localization" becomes "l10n"
 * "internationalization" becomes "i18n"
 * "word" remains "word"
 *
 * @algorithm
 * 1. Read the number of words to process, t.
 * 2. Loop t times:
 *    a. Read the input word.
 *    b. Check if the word's length is greater than 10.
 *    c. If it is, calculate the length of the middle part (length - 2) and
 *       print the first character, the calculated length, and the last character.
 *    d. If not, print the word as is.
 *
 * @time_complexity O(T * L), where T is the number of words and L is the max
 *   length of a word (due to reading the input). The processing for each word
 *   is O(1).
 * @space_complexity O(L) to store the longest word.
 */
#include <iostream>
#include <string>
using namespace std;

int main() {
  int t, l, i;
  string a;
  cin >> t;
  // Block Logic: Process each of the 't' test cases.
  while (t--) {
    cin >> a;
    // Pre-condition: Check if the word's length exceeds the threshold for abbreviation.
    if (a.length() > 10) {
      // Block Logic: Abbreviate the word.
      l = a.length() - 2;
      // The last character is at index a.length() - 1, which is equivalent to l + 1.
      cout << a[0] << l << a[l + 1] << "\n";
    } else {
      // Block Logic: If the word is not too long, print it without modification.
      cout << a << "\n";
    }
  }

  return 0;
}
