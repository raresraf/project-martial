/**
 * @file 71A.CPP
 * @brief Abbreviates words that are "way too long".
 * @details This program is a solution to the "Way Too Long Words" problem from
 * Codeforces (71A). It reads a list of words and for any word that is longer
 * than 10 characters, it prints an abbreviation. Otherwise, it prints the
 * word as is. The abbreviation consists of the first character, the number of
 * characters in between, and the last character.
 *
 * @algorithm Conditional abbreviation based on string length.
 * @complexity
 * Time: O(N * L), where N is the number of words and L is the maximum length
 *       of a word. This is dominated by the time taken to read the input strings.
 *       The processing for each string is O(1).
 * Space: O(L) to store the longest word.
 */
#include <iostream>
#include <string>

using namespace std;

/**
 * @brief Main function to handle word processing and abbreviation.
 */
int main() {
  // n: the number of words to process.
  int n;
  cin >> n;

  // Pre-condition: 'n' specifies the number of test cases.
  // Block Logic: Loop 'n' times to process each word.
  while (n--) {
    string s;
    cin >> s;

    // Check if the word's length is greater than 10.
    if (s.length() > 10)
      // If so, print the abbreviation: first char, length-2, last char.
      // e.g., "localization" becomes "l10n".
      cout << s[0] << s.length() - 2 << s[s.length() - 1] << endl;
    else
      // Otherwise, print the word unchanged.
      cout << s << endl;
  }

  return 0;
}
