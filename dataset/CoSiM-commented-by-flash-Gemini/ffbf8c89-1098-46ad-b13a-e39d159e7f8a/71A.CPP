/**
 * @file 71A.CPP
 * @brief Solution to a problem that abbreviates long words.
 *
 * This program reads an integer 't' representing the number of words to process.
 * For each word, if its length is greater than 10, it is abbreviated to its first character,
 * followed by the count of intermediate characters, and then its last character.
 * Otherwise, the word is printed as is.
 *
 * Algorithm: Iterative processing of strings with conditional abbreviation based on length.
 * Time Complexity: O(T * L), where T is the number of test cases and L is the maximum string length.
 * Space Complexity: O(L) for storing the input string.
 */
#include <iostream>
#include <string> // Include for string manipulation
using namespace std;
int main() {
  // Declare 't' for the number of test cases and 's' for storing the input string.
  int t;
  string s;
  // Read the total number of test cases 't'.
  cin >> t;
  /**
   * Block Logic: Loop through each test case, processing one word per iteration.
   * Invariant: 't' is decremented, and one word is read and potentially abbreviated.
   */
  while (t--) {
    // Read the current word.
    cin >> s;
    /**
     * Block Logic: Checks the length of the word to decide whether to abbreviate it.
     * Pre-condition: 's' contains the word to be processed.
     */
    if (s.size() <= 10)
      // If the word's length is 10 or less, print it as is.
      cout << s << endl;
    else
      // If the word's length is greater than 10, print its abbreviated form:
      // first character, count of middle characters, last character.
      cout << s[0] << s.size() - 2 << s[s.size() - 1] << endl;
  }
  return 0;
}
