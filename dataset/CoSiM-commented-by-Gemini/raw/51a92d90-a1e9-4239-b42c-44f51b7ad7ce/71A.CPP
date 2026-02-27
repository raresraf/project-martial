/**
 * @file 71A.CPP
 * @brief Solves the "Way Too Long Words" problem from Codeforces.
 * @details This program abbreviates words that are longer than 10 characters.
 * The abbreviation consists of the first character, the number of characters
 * between the first and last, and the last character. Words with 10 or fewer
 * characters are printed without modification.
 *
 * @algorithm
 * 1. Read the number of words to process, n.
 * 2. Loop n times:
 *    a. Read the word into a string `s`.
 *    b. If the length of `s` is greater than 10:
 *       i. Print the first character using `s.front()`.
 *       ii. Print the length of the string minus 2.
 *       iii. Print the last character using `s.back()`.
 *    c. Otherwise, print the string `s` as is.
 *
 * @complexity
 * Time Complexity: O(n * L), where n is the number of words and L is the
 * maximum length of a word, due to the time taken to read each string.
 * The processing for each string is O(1) after it's read.
 * Space Complexity: O(L) to store the longest word.
 */
#include <iostream>
#include <string>
using namespace std;

int main() {
  int n; // Number of words to process.
  cin >> n;

  /**
   * @block
   * @brief Process each word.
   */
  while (n--) {
    string s;
    cin >> s;

    /**
     * @brief Check if the word's length is greater than 10.
     */
    if (s.size() <= 10) {
      // If not, print the word as is.
      cout << s << endl;
    } else {
      // If it is, print the abbreviated form.
      // s.front() gets the first character.
      // s.back() gets the last character.
      cout << s.front() << s.size() - 2 << s.back() << endl;
    }
  }
}
