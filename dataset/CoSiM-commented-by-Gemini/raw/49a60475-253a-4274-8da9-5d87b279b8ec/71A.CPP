/**
 * @file 71A.CPP
 * @brief Solution to Codeforces problem 71A (Way Too Long Words).
 * @author Solved by: [User who submitted the code]
 *
 * @details
 * This program processes a list of words. For each word, if its length is
 * greater than 10 characters, it is replaced with a special abbreviation.
 * Otherwise, the word is printed as is.
 *
 * @section problem_description
 * The abbreviation consists of the first letter of the word, followed by the
 * number of letters between the first and the last one, and finally the last
 * letter of the word. For example, "localization" would be abbreviated as "l10n",
 * and "internationalization" would be "i18n".
 *
 * @section algorithm
 * 1. Read an integer `n`, the number of words to be processed.
 * 2. Loop `n` times to handle each word.
 * 3. Inside the loop, read a word into a string variable.
 * 4. Check the length of the string.
 * 5. If the length is strictly greater than 10, construct and print the
 *    abbreviation:
 *    - The first character (`word[0]`).
 *    - The length of the word minus 2.
 *    - The last character (`word[word.length() - 1]`).
 * 6. If the length is 10 or less, print the word unchanged.
 *
 * @section complexity
 * - Time Complexity: O(N * L), where N is the number of words and L is the
 *   maximum length of a word. This is dominated by the string input operations.
 * - Space Complexity: O(L) to store the longest word.
 */

#include <bits/stdc++.h>
using namespace std;

int main() {
  // n: The number of words to process.
  int n;
  cin >> n;

  string a; // Variable to store each word.

  /**
   * @brief Loop to process each of the n words.
   */
  for (int i = 0; i < n; i++) {
    cin >> a;

    // Pre-condition: 'a' contains a word.
    // Check if the word is "too long" (length > 10).
    if (a.length() > 10) {
      // If so, print the abbreviation.
      // a[0]: first character.
      // a.length() - 2: number of internal characters.
      // a[a.length() - 1]: last character.
      cout << a[0] << a.length() - 2 << a[a.length() - 1] << endl;
    } else {
      // Otherwise, print the word as is.
      cout << a << endl;
    }
  }

  return 0;
}
