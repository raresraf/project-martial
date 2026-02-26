/**
 * @file 71A.CPP
 * @brief Solves the "Way Too Long Words" problem.
 * @details This program implements a common string abbreviation algorithm often seen
 * in competitive programming. For a given list of words, it shortens any word
 * that is longer than 10 characters. The abbreviated form consists of the
 * word's first character, followed by the count of characters between the first
 * and last, and finally the last character. Words with 10 or fewer characters
 * are left unchanged.
 *
 * @algorithm String length check and conditional abbreviation.
 * @complexity
 * Time Complexity: O(T * L_max), where T is the number of test cases (words)
 * and L_max is the maximum length of a word. The primary operations are reading
 * the string and calculating its length.
 * Space Complexity: O(L_max) to store the longest input string.
 */

// A common but non-standard header in competitive programming that includes
// most of the standard library. For production code, it is better to include
// specific headers like <iostream> and <string>.
#include <bits/stdc++.h>

// Brings all names from the std namespace into the current scope,
// avoiding the need to prefix standard library components with `std::`.
// This is common in competitive programming but can be problematic in larger projects.
using namespace std;

int main() {
  // Variable `t` holds the number of words to be processed.
  int t;
  cin >> t;

  string temp; // A temporary string to hold each word read from input.

  /**
   * @block
   * @brief Main processing loop for all test cases.
   * @invariant The loop runs `t` times, processing one word in each iteration.
   */
  for (int i = 0; i < t; ++i) {
    cin >> temp;
    int l = temp.length(); // Get the length of the current word.

    /**
     * @block
     * @brief Conditional abbreviation logic.
     * @pre The word's length `l` has been computed.
     * @post The word is printed, either in its original or abbreviated form.
     */
    if (l <= 10) {
      // If the word's length is 10 or less, print it as is.
      cout << temp << endl;
    } else {
      // If the word is longer than 10 characters, abbreviate it.
      // The format is: first character, number of middle characters, last character.
      cout << temp[0] << l - 2 << temp[l - 1] << endl;
    }
  }

  return 0;
}
