/**
 * @file 71A.CPP
 * @brief Solution for Codeforces Problem 71A, "Way Too Long Words".
 * @details This program abbreviates words that are longer than 10 characters.
 * The implementation correctly follows the problem specification. It uses a mix
 * of C-style (`scanf`) and C++-style (`cin`, `cout`, `string`) I/O.
 *
 * Algorithm:
 * 1. Read the number of words to process, `n`.
 * 2. Loop `n` times.
 * 3. In each iteration, read an input word.
 * 4. Check if the length of the word is strictly greater than 10.
 * 5. If it is, print the abbreviation: the first character, the count of the
 *    middle characters (length - 2), and the last character.
 * 6. If the length is 10 or less, print the word unchanged.
 *
 * Time Complexity: O(N * L), where N is the number of words and L is their
 * maximum length, due to string input operations.
 * Space Complexity: O(L) to store each word.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  string s;
  scanf("%d", &n);
  /**
   * @brief This loop iterates `n` times to process each word.
   */
  while (n--) {
    cin >> s;
    /**
     * @brief Implements the core logic: if the word length exceeds 10,
     * abbreviate it; otherwise, print it as is.
     */
    if (s.length() > 10)
      cout << s[0] << s.length() - 2 << s[s.length() - 1] << endl;
    else
      cout << s << endl;
  }
}
