/**
 * @file 71A.CPP
 * @brief Abbreviates words that are too long.
 *
 * This program reads a list of words and shortens any word that has a length
 * strictly greater than 10. The abbreviation consists of the first character,
 * the number of characters between the first and last, and the last character.
 * Words with 10 or fewer characters are left unchanged.
 *
 * Algorithm:
 * 1. Read an integer `n`, the number of words to process.
 * 2. Loop `n` times.
 * 3. In each iteration, read a word into a string `s`.
 * 4. Get the length of the string, `len`.
 * 5. If `len` is greater than 10, print the first character `s[0]`, followed by
 *    the integer `len - 2`, followed by the last character `s[len - 1]`.
 * 6. Otherwise (if `len` is 10 or less), print the string `s` as is.
 *
 * Time Complexity: O(N * L), where N is the number of words and L is the
 * maximum length of a word. Reading each string takes O(L) time.
 * Space Complexity: O(L) to store the longest word read.
 */
#include <iostream>
#include <string>

using namespace std;

int main() {
  // Fast I/O optimization
  ios::sync_with_stdio(0);
  cin.tie(0);

  int n;
  // Read the number of test cases.
  cin >> n;

  /**
   * @brief Main loop to process each word.
   * Pre-condition: `n` is the number of words to process.
   */
  for (int i = 0; i < n; i++) {
    string s;
    cin >> s;

    int len = s.length();

    /**
     * @brief Check if the word needs to be abbreviated.
     * Block Logic: If the length is over 10, print the abbreviation,
     * otherwise print the word itself.
     */
    if (len > 10) {
      cout << s[0] << len - 2 << s[len - 1] << endl;
    } else {
      cout << s << endl;
    }
  }

  return 0;
}
