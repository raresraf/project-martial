/**
 * @file 71A.CPP
 * @brief Abbreviates words that are "too long".
 *
 * This program solves the "Way Too Long Words" problem from Codeforces. It reads
 * a list of words and for each word that has more than 10 characters, it replaces
 * it with a special abbreviation. The abbreviation consists of the first letter,
 * the number of letters in between, and the last letter.
 *
 * @algorithm String Manipulation
 * @complexity
 * - Time: O(N * L), where N is the number of words and L is the maximum length of a word.
 *   In practice, string reading and length calculation are the dominant factors.
 * - Space: O(L) to store the longest word.
 */
#include <iostream>
#include <string>

using namespace std;

int main() {
  // n: the number of words to process.
  int n;
  // s: stores each word read from input.
  string s;
  cin >> n;

  /**
   * @brief Block Logic: Processes each of the `n` words.
   * Invariant: The loop executes `n` times, processing one word per iteration.
   */
  while (n--) {
    cin >> s;
    /**
     * @brief Pre-condition: Checks if the length of the word exceeds 10 characters.
     */
    if (s.length() > 10) {
      /**
       * @brief Logic: Constructs the abbreviation.
       * It consists of the first character, the count of the middle characters
       * (which is length - 2), and the last character.
       */
      cout << s[0] << s.length() - 2 << s[s.length() - 1] << "\n";
    } else {
      // If the word is not too long, print it as is.
      cout << s << "\n";
    }
  }
  return 0;
}
