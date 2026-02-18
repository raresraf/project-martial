/**
 * @file 71A.CPP
 * @brief Solution to the "Way Too Long Words" problem (71A) from Codeforces.
 * @details This program shortens words that are strictly longer than 10 characters
 * into a special abbreviation. The abbreviation consists of the first letter,
 * the number of letters between the first and last, and the last letter.
 * For example, "localization" becomes "l10n".
 *
 * @algorithm The solution iterates through a given number of words, applying the
 * abbreviation rule to each.
 * 1. Read the total number of words (n).
 * 2. Loop 'n' times to process each word.
 * 3. In each iteration, read the word.
 * 4. Check if the word's length is greater than 10.
 * 5. If it is, construct the abbreviation. The code uses `string::replace` for an
 *    in-place modification, replacing the middle characters with the number of
 *    letters between the first and last.
 * 6. Print the resulting word (either the original or the abbreviation).
 *
 * @complexity
 * Time Complexity: O(N * L), where N is the number of words and L is the maximum length of a word.
 * The complexity is dominated by the I/O operations for reading and writing the strings.
 * Space Complexity: O(L), for storing the longest word.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  int n;
  cin >> n;
  /**
   * @block
   * @brief Processes 'n' words, one per iteration.
   * @pre 'n' is the number of words to process.
   * @post Each of the 'n' words has been read, processed, and printed.
   */
  while (n--) {
    string s;
    cin >> s;
    /**
     * @block
     * @brief Checks if a word is "too long" and abbreviates it if necessary.
     * @pre 's' contains the word to process.
     * @post If 's' was longer than 10 characters, it is replaced by its abbreviation.
     *       Otherwise, it remains unchanged.
     */
    if (s.length() > 10)
      // Replace the substring from the second character (index 1) with a length of (total_length - 2)
      // with the string representation of that length.
      s.replace(1, s.length() - 2, to_string(s.length() - 2));
    cout << s << endl;
  }
}