
#include <cstdio>
#include <iostream>
#include <string>

using namespace std;

/**
 * @file 71A.CPP
 * @brief Solution to Codeforces problem 71A, "Way Too Long Words".
 * @details This program abbreviates words that are longer than 10 characters.
 * The abbreviation consists of the first letter, the count of characters
 * between the first and last letters, and the last letter.
 *
 * Algorithm:
 * For each input word, the program checks its length. If the length exceeds 10,
 * it constructs and prints the abbreviation. Otherwise, it prints the word
 * as is.
 *
 * Implementation Note:
 * The input reading loop is flawed. It runs `n+1` times, and the use of
 * `getline` after `cin >> n` will consume the trailing newline from the first
 * input, causing it to misprocess the first word.
 *
 * Time Complexity: O(N * L), where N is the number of words and L is their
 * maximum length. The `to_string` operation can take time proportional to the
* number of digits in the length.
 * Space Complexity: O(L) to store the longest word.
 */
int main() {
  int i, n;
  cin >> n;

  /**
   * @brief Main processing loop for each word.
   *
   * Flaw: This loop runs `n+1` times, not `n`. The first `getline` will read
   * the empty line left by `cin >> n`, causing issues with input handling.
   */
  for (i = 0; i <= n; i++) {
    string s;

    getline(cin, s);
    /**
     * @brief Abbreviation logic block.
     *
     * Pre-condition: `s` contains a line of input.
     * Logic: If the string length is over 10, it's abbreviated. Otherwise,
     * it's printed as is. An empty string (from the first flawed getline)
     * will be printed as a blank line.
     */
    if (s.size() <= 10) {
      cout << s << endl;
    } else if (s.size() > 10) {
      // Construct the abbreviation: first char + (length-2) + last char.
      string s3 = s[0] + to_string(s.size() - 2) + s[s.size() - 1];
      cout << s3 << endl;
    }
  }
  return (0);
}
