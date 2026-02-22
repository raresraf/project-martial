/**
 * @file 71A.CPP
 * @brief Solution for the "Way Too Long Words" problem (Codeforces 71A).
 * @see http://codeforces.com/problemset/problem/71/A
 *
 * @algorithm
 * The problem requires shortening words that are longer than 10 characters into
 * an abbreviation. The abbreviation consists of the word's first character,
 * the number of letters between the first and last characters, and the last
 * character. Words with 10 or fewer characters are not changed.
 *
 * The algorithm directly implements these rules:
 * 1. Read the number of words, `n`.
 * 2. Loop `n` times.
 * 3. In each iteration, read the word and get its length.
 * 4. If the length is greater than 10, print the constructed abbreviation.
 * 5. Otherwise, print the word as is.
 *
 * @complexity
 * - Time Complexity: O(N * L), where N is the number of words and L is the
 *   maximum length of a word, dominated by the input reading process.
 * - Space Complexity: O(L) to store the longest input word.
 */
// http://codeforces.com/problemset/problem/71/A

#include <iostream>
#include <string> // Recommended for std::string
using namespace std;

int main() {
  int n;  // The number of words to process.
  int le; // Variable to store the length of a word.
  string s; // Variable to store the word itself.
  cin >> n;
  /**
   * @brief Main loop to process each of the `n` words.
   */
  while (n--) {
    cin >> s;
    le = s.length();
    /**
     * @brief Check if the word is longer than 10 characters.
     */
    if (le > 10)
      /**
       * @brief If the word is too long, print its abbreviation:
       * - The first character: s[0]
       * - The number of characters between: le - 2
       * - The last character: s[le - 1]
       */
      cout << s[0] << le - 2 << s[le - 1] << endl;
    else
      // If the word is not too long, print it unchanged.
      cout << s << endl;
  }
  return 0;
}
