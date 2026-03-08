/**
 * @file 71A.CPP
 * @brief Solution for the "Way Too Long Words" problem.
 *
 * @details
 * This program processes a list of words. For each word, it checks its length.
 * If the word is longer than 10 characters, it is replaced by a special
 * abbreviation. The abbreviation consists of the first character of the word,
 * followed by the number of characters between the first and last character,
 * and finally the last character of the word. Words with 10 or fewer characters
 * are printed without modification.
 *
 * @section
 * Algorithm: Conditional string manipulation based on length.
 * Time Complexity: O(N), where N is the number of words, as each word is processed once. The operations on each word are O(1).
 * Space Complexity: O(L_max), where L_max is the length of the longest word.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {

  // n: the number of words to process.
  int n;
  cin >> n;

  /**
   * @brief This loop processes each of the 'n' words.
   */
  for (int i = 0; i < n; i++) {

    // 'a' stores the current word.
    string a;
    cin >> a;

    /**
     * @brief Conditional logic to either print the word or its abbreviation.
     */
    // Pre-condition: Check if the word's length exceeds the threshold of 10.
    if (a.length() <= 10) {
      // If the word is not "too long", print it as is.
      cout << a << endl;
      ; // This is a redundant, empty statement.
    } else {
      /**
       * @brief Constructs and prints the abbreviation for a long word.
       *
       * Logic: The abbreviation is formed by:
       * - The first character: a[0]
       * - The count of middle characters: a.length() - 2
       * - The last character: a[a.length() - 1]
       */
      cout << a[0] << a.length() - 2 << a[a.length() - 1] << endl;
    }
  }
}
