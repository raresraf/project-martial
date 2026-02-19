/**
 * @file 71A.CPP
 * @brief Solves the "Way Too Long Words" problem by abbreviating long strings.
 * @details This program reads a number of words and, for each word, checks if its
 * length is greater than 10. If it is, the word is replaced by a special
 * abbreviation. Otherwise, the word is printed as is.
 *
 * @algorithm The algorithm iterates through each word. For each word, it checks its
 * length. If the length exceeds 10, it constructs an abbreviation consisting of the
 * first character, the decimal representation of the number of characters between
 * the first and the last (length - 2), and the last character.
 *
 * @time_complexity O(N), where N is the total number of characters across all input words.
 * This is because reading the input is the dominant operation.
 * @space_complexity O(L_max), where L_max is the maximum length of any single word,
 * required to store the word in memory.
 */
#include <iostream>
#include <string>
using namespace std;
int main() {
  string s;
  int t;
  cin >> t;
  /**
   * Block Logic: Process 't' number of words, applying the abbreviation logic to each.
   * Invariant: The loop will execute 't' times, processing one word per iteration.
   */
  while (t--) {
    cin >> s;
    int l = s.length();
    /**
     * Block Logic: Conditionally abbreviate the word based on its length.
     * Pre-condition: 's' contains the input word and 'l' its length.
     */
    if (l > 10) {
      // Logic: If the word is longer than 10 characters, print its abbreviation.
      // The abbreviation is the first char, the count of intermediate chars, and the last char.
      cout << s[0] << l - 2 << s[l - 1] << endl;
    } else {
      // Logic: If the word is 10 characters or shorter, print it unchanged.
      cout << s << endl;
    }
  }
  return 0;
}
