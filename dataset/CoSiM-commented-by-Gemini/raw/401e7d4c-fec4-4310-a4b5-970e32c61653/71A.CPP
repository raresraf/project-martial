/**
 * @file 71A.CPP
 * @brief Solution to Codeforces problem 71A - Way Too Long Words.
 *
 * @algorithm
 * The problem requires abbreviating words that are "too long". A word is considered
 * too long if its length is strictly greater than 10. The abbreviation consists
 * of the first character, followed by the number of characters between the first
 * and last characters, and finally the last character. Words with a length of 10
 * or less should remain unchanged.
 *
 * The algorithm is as follows:
 * 1. Read the number of words, `n`.
 * 2. Loop `n` times to process each word.
 * 3. In each iteration, read the word.
 * 4. Check if the word's length is greater than 10.
 * 5. If it is, construct and print the abbreviation (e.g., "localization" becomes "l10n").
 * 6. If not, print the word as is.
 *
 * @note The `<bits/stdc++.h>` header is a non-standard convenience header, and
 * `<math.h>` is not used. The variable `temp` is declared but never used.
 *
 * @complexity
 * - Time Complexity: O(N * L_max), where N is the number of words and L_max is the
 *   maximum length of a word. This is dominated by the time to read the input strings.
 * - Space Complexity: O(L_max) to store the longest word.
 */
#include <bits/stdc++.h>
#include <iostream>
#include <math.h>
using namespace std;

int main() {
  int n, temp;
  cin >> n;
  temp = n; // This variable is unused.
  
  /**
   * @brief This loop iterates `n` times to process each word.
   */
  while (n--) {
    string a;
    cin >> a;
    int l = a.length();
    
    /**
     * @brief Pre-condition: Check if the word length exceeds 10.
     * Invariant: Based on the length, either abbreviate the word or print it directly.
     */
    if (l > 10) {
      // Logic for abbreviating long words.
      int lc = l - 2; // Number of characters between the first and last.
      cout << a.at(0) << lc << a.at(l - 1) << endl;
    } else {
      // Logic for words that are not too long.
      cout << a << endl;
    }
  }
  return 0;
}
