/**
 * @file 71A.CPP
 * @brief Solution to a problem that abbreviates long words.
 *
 * This program reads an integer 't' representing the number of words to process.
 * For each word, if its length is greater than 10 (i.e., length >= 11), it is
 * abbreviated to its first character, followed by the count of intermediate characters,
 * and then its last character. Otherwise, the word is printed as is.
 * This is a common string manipulation task often encountered in competitive programming.
 *
 * Algorithm: Iterative processing of strings with conditional abbreviation based on length.
 * Time Complexity: O(T * L), where T is the number of words and L is the maximum string length.
 * Space Complexity: O(L) for storing the input string.
 */
#include <bits/stdc++.h> // Includes various standard libraries like iostream, string.
using namespace std;     // Using the standard namespace.

int main() {
  // Declare 't' for the number of test cases.
  int t;
  // Read the total number of test cases 't'.
  cin >> t;
  /**
   * Block Logic: Loop through each test case, processing one word per iteration.
   * Invariant: 't' is decremented, and one word is read and either its full form
   *            or abbreviated form is printed.
   */
  while (t--) {
    string s; // Declare a string variable to store the input word for the current test case.
    // Read the current word.
    cin >> s;
    /**
     * Block Logic: Checks the length of the current word to decide on abbreviation.
     * Pre-condition: 's' contains the word to be processed.
     */
    if (s.length() < 11)
      // If the word's length is 10 or less, print it as is.
      cout << s << endl;
    else
      // If the word's length is greater than 10 (>=11), print its abbreviated form:
      // first character, count of middle characters (length - 2), last character.
      cout << s[0] << s.length() - 2 << s.back() << endl;
  }

  return 0; // Indicate successful program execution.
}