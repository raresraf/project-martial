/**
 * @file 71A.CPP
 * @brief Solves the "Way Too Long Words" problem by abbreviating long strings.
 *
 * This program reads a number of words and, for each word longer than 10
 * characters, it creates an abbreviation. The abbreviation consists of the
 * first character, the number of middle characters, and the last character.
 * For example, "internationalization" becomes "i18n".
 *
 * Algorithm:
 * 1. Read the number of test cases (t).
 * 2. Loop 't' times:
 *    a. Read the input word.
 *    b. Get the length of the word.
 *    c. If the length is greater than 10, print the abbreviation.
 *    d. Otherwise, print the word as is.
 *
 * Time Complexity: O(T * L), where T is the number of test cases and L is the
 * length of the word. Since L is typically small, this is effectively O(T).
 * Space Complexity: O(L_max) to store the longest word read.
 */
#include <iostream>
#include <string> // Required for std::string and std::to_string

using namespace std;

int main() {
  int t; // Number of test cases
  cin >> t;

  string s; // Holds the word for each test case

  /**
   * Block Logic: Main processing loop.
   * This loop iterates for the number of test cases specified.
   */
  while (t--) {
    cin >> s;

    int len = s.length();

    /**
     * Block Logic: Abbreviation conditional.
     * Checks if the word's length exceeds the threshold of 10 characters.
     */
    if (len <= 10) {
      // If the word is short enough, print it without modification.
      cout << s << endl;
    } else {
      // If the word is too long, construct and print the abbreviation.
      // s[0]: first character.
      // to_string(len - 2): the count of characters between the first and last.
      // s[len - 1]: last character.
      cout << s[0] << to_string(len - 2) << s[len - 1] << endl;
    }
  }

  return 0;
}
