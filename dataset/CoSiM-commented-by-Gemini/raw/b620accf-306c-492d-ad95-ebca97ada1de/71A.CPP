/**
 * @file 71A.CPP
 * @brief Solution to the "Way Too Long Words" problem (Codeforces 71A).
 *
 * @details
 * This program processes a list of words. For each word, it checks if its length is
 * greater than 10 characters.
 * - If the word is longer than 10 characters, it is replaced by a special abbreviation.
 *   This abbreviation consists of the first character of the word, followed by the
 *   number of characters between the first and last characters, and concluding with
 *   the last character of the word. For example, "localization" becomes "l10n".
 * - If the word is 10 characters or shorter, it is printed as is.
 *
 * The program first reads the number of words, `n`, and then processes each of the
 * `n` words according to this rule.
 *
 * Algorithm:
 * 1. Read the number of words, `n`.
 * 2. Loop `n` times:
 *    a. Read a word.
 *    b. If its length is greater than 10, print its abbreviation.
 *    c. Otherwise, print the word itself.
 *
 * Time Complexity: O(N), where N is the number of words to process. The operations
 *                  inside the loop are effectively constant time as string lengths are small.
 * Space Complexity: O(L_max), where L_max is the maximum length of a word, needed to store it.
 */
#include <bits/stdc++.h>
#include <string>
using namespace std;

int main() {
  string s;
  int n, a;

  cin >> n;

  // The outer if is redundant as problem constraints are guaranteed by the judge.
  if (n >= 1 && n <= 100) {
    /**
     * @brief Main loop to process each word.
     * @details Iterates `n` times, reading one word in each iteration and applying the
     * abbreviation logic.
     */
    for (int i = 0; i < n; i++) {
      cin >> s;
      /**
       * @brief Checks and applies the abbreviation rule.
       * @details If the word's length exceeds 10, it prints the abbreviation in the
       * format: [first letter][length-2][last letter]. Otherwise, it prints
       * the original word.
       */
      if (s.length() > 10) {
        cout << s[0] << s.length() - 2 << s[s.length() - 1] << endl;
      } else
        cout << s << endl;
    }
  }
}