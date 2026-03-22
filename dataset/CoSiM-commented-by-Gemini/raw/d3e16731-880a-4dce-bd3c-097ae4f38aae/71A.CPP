/**
 * @file 71A.CPP
 * @brief Solves the "Way Too Long Words" problem from Codeforces (Problem 71A).
 * @details This program reads a series of words. If a word is longer than 10
 * characters, it is shortened to an abbreviation; otherwise, it is printed as is.
 *
 * @section algorithm
 * Algorithm: Conditional String Manipulation
 * 1. Read the number of words, N.
 * 2. Loop N times:
 *    a. Read a word.
 *    b. Check if its length is greater than 10.
 *    c. If true, print the first character, the count of the middle characters
 *       (length - 2), and the last character.
 *    d. If false, print the original word.
 *
 * @section complexity
 * - Time Complexity: O(N * L_avg), where N is the number of words and L_avg is
 *   their average length. The complexity is linear in the total size of the input.
 * - Space Complexity: O(L_max), to store the longest word read from the input.
 */
#include <iostream>
#include <string>
using namespace std;
int main() {
  int n, i, l;
  string s;
  cin >> n;

  /**
   * @brief This loop processes each of the `n` words.
   */
  for (i = 0; i < n; i++) {
    cin >> s;

    /**
     * @brief Checks if the word is "too long" and prints the appropriate format.
     */
    if (s.length() > 10) {
      // The abbreviation consists of the first character, the number of
      // internal characters (length - 2), and the last character.
      cout << s[0] << s.length() - 2 << s[s.length() - 1] << endl;
    } else {
      cout << s << endl;
    }
  }

  return 0;
}
