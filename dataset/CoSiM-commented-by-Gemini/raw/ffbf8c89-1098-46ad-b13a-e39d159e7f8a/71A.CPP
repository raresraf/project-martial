/**
 * @file 71A.CPP
 * @brief Solves the "Way Too Long Words" problem on Codeforces.
 *
 * @details This program processes a series of words. If a word's length is
 * greater than 10, it is abbreviated. Otherwise, it is printed as is.
 *
 * Algorithm:
 * 1. Read the number of test cases, 't'.
 * 2. Loop 't' times:
 *    a. Read a word as a string 's'.
 *    b. If the length of 's' is 10 or less, print 's' directly.
 *    c. If the length is greater than 10, print the abbreviation, which consists
 *       of the first character, the number of characters between the first and
 *       last (i.e., length - 2), and the last character.
 *
 * Time Complexity: O(T * L) for reading T strings of max length L. The processing
 * for each string is O(1).
 * Space Complexity: O(L) to store the longest string.
 */
#include <iostream>
using namespace std;
int main() {
  int t;
  string s;
  cin >> t;
  /**
   * @brief Main loop to process each word.
   * Pre-condition: 't' holds the number of words to process.
   * Invariant: The loop executes 't' times, processing one word per iteration.
   */
  while (t--) {
    cin >> s;
    // Check if the word needs to be abbreviated.
    if (s.size() <= 10)
      cout << s << endl;
    else
      // Print the abbreviated form.
      cout << s[0] << s.size() - 2 << s[s.size() - 1] << endl;
  }
  return 0;
}
