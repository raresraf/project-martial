/**
 * @file 71A.CPP
 * @brief Solution to the "Way Too Long Words" problem (71A) from Codeforces.
 * @details This program shortens words that are strictly longer than 10 characters
 * by replacing the middle letters with a count. For example, "localization" becomes "l10n".
 *
 * @author Your Name (or leave blank)
 *
 * @algorithm This solution directly constructs and prints the output for each word.
 * 1. Read the number of words to process, `t`.
 * 2. Loop `t` times.
 * 3. In each iteration, read the word `s`.
 * 4. If the length of `s` is greater than 10:
 *    a. Print the first character of `s`.
 *    b. Print the integer value of `length - 2`.
 *    c. Print the last character of `s`.
 * 5. Otherwise (if the word is not too long), print the original word `s`.
 *
 * @note This implementation uses `main()` without a return type, which is non-standard C++.
 *
 * @complexity
 * Time Complexity: O(T * L), where T is the number of words and L is the maximum length
 * of a word. The complexity is dominated by I/O operations.
 * Space Complexity: O(L), for storing the longest input word.
 */
#include <bits/stdc++.h>
using namespace std;
// Using `main()` without an explicit `int` return type is non-standard in C++.
main() {
  ios::sync_with_stdio(0);
  cin.tie(0);
  int t;
  cin >> t;
  /**
   * @block
   * @brief Processes 't' words, one per iteration.
   */
  while (t--) {
    string s;
    cin >> s;
    /**
     * @block
     * @brief Abbreviates the word if its length is > 10, otherwise prints it as is.
     */
    if (s.length() > 10) {
      cout << s[0] << s.length() - 2 << s[s.length() - 1] << "\n";

    }

    else {
      cout << s << "\n";
    }
  }
}
