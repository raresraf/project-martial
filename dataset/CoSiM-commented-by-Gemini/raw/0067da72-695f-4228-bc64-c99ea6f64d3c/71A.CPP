/**
 * @file 71A.CPP
 * @brief Abbreviates long words according to specific rules.
 * @note Link to Problem: https://codeforces.com/problemset/problem/71/A
 *
 * This program processes a series of words. If a word's length is greater than 10,
 * it is abbreviated to a format consisting of its first letter, the count of
 * characters between the first and last letters, and its last letter.
 * Otherwise, the word is printed as is.
 *
 * @b Algorithm: The core logic involves checking the length of each input string.
 * If the length exceeds 10, an abbreviated string is constructed and printed.
 * Otherwise, the original string is printed.
 *
 * @b Time_Complexity: O(N*L), where N is the number of words and L is the maximum length of a word.
 * In practice, string length calculation and character access are very fast.
 * @b Space_Complexity: O(L), where L is the maximum length of a word, to store each input string.
 */
// Link to Problem: https://codeforces.com/problemset/problem/71/A

#include <iostream>

using namespace std;

int main() {
  int n;
  // Read the number of words to process.
  cin >> n;
  /**
   * @brief This loop processes each word.
   *
   * Pre-condition: 'n' is the number of words to be processed.
   * Invariant: After each iteration, one word has been processed and printed
   * (either in its original or abbreviated form).
   */
  while (n--) {
    string s;
    // Read the next word.
    cin >> s;
    int si = s.length();
    // Block Logic: If the word is "too long" (length > 10), abbreviate it.
    if (si > 10) {
      // The abbreviation is the first character, the number of middle characters, and the last character.
      cout << s[0] << si - 2 << s[si - 1] << '\n';
    } else
      // Otherwise, print the word as is.
      cout << s << '\n';
  }

  return 0;
}
