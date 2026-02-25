/**
 * @file 71A.CPP
 * @brief Solution for a string abbreviation problem ("Way Too Long Words").
 * @details This program abbreviates words that are longer than 10 characters.
 * The abbreviation consists of the first letter, the number of letters between
 * the first and last, and the last letter. For example, "internationalization"
 * becomes "i18n". Words with 10 or fewer characters are not changed.
 *
 * Algorithm:
 * 1. Read the number of words to process, 'n'.
 * 2. Loop 'n' times for each word.
 * 3. Inside the loop, read the word.
 * 4. If the word's length is greater than 10:
 *    a. Construct the abbreviation by concatenating the first character,
 *       the string representation of (length - 2), and the last character.
 *    b. Print the abbreviation.
 * 5. Otherwise, print the original word.
 *
 * Time Complexity: O(N * L_avg), where N is the number of words and L_avg is their average length.
 * Space Complexity: O(L_max), where L_max is the length of the longest word.
 */
#include <bits/stdc++.h>

using namespace std;

int main(void) {
  int n;
  cin >> n;

  /**
   * @block
   * @description This loop iterates 'n' times to process each word provided in the input.
   */
  while (n--) {
    string s;
    cin >> s;

    /**
     * @block
     * @description Checks if the word's length exceeds 10 characters to determine
     * if abbreviation is necessary.
     */
    if (s.length() > 10) {
      // Create a string for the number of middle characters.
      std::string l = std::to_string(s.length() - 2);
      // Print the abbreviated word, e.g., "l" + "10" + "n".
      cout << (s[0] + l + s[s.length() - 1]) << endl;
    } else {
      // If the word is not too long, print it as is.
      cout << s << endl;
    }
  }
}
