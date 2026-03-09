/**
 * @file 71A.CPP
 * @brief Solution to Codeforces Problem 71A: "Way Too Long Words" using C++ STL.
 *        This program processes a series of words, abbreviating those
 *        that are excessively long (length > 10) by replacing their
 *        middle characters with a count, while printing shorter words as is.
 *
 * Algorithm: String handling with C++ string class, length checking, and conditional formatting.
 *            For words longer than 10 characters, it extracts the first character,
 *            calculates the count of intermediate characters (length - 2), and appends
 *            the last character.
 * Time Complexity: O(L) for each word, where L is the length of the word,
 *                  due to string input and basic string operations. Total
 *                  time complexity is O(N * L_max) for N test cases.
 * Space Complexity: O(L_max) to store the longest input word.
 */

#include <iostream>
#include <string>
using namespace std;

int main() {
  /**
   * Functional Utility: Manages the execution flow for multiple test cases,
   *                     reading input words and applying the abbreviation logic.
   */
  int n; // Declares an integer to store the number of test cases.
  cin >> n; // Reads the number of test cases from standard input.

  /**
   * Block Logic: Processes each test case by reading a word and applying the abbreviation logic.
   * Pre-condition: `n` holds the number of remaining test cases.
   * Invariant: The loop continues until all `n` test cases are processed.
   */
  while (n--) {
    string s; // Declares a string to store the current word for processing.
    cin >> s; // Reads the word for the current test case into string `s`.

    /**
     * Block Logic: Checks the length of the current word and applies conditional formatting.
     * Pre-condition: `s` contains the word.
     * Invariant: Either the original word or its abbreviated form is printed.
     */
    if (s.size() <= 10) {
      cout << s << endl; // Prints the word as is if its length is 10 or less.
    } else {
      // Prints the abbreviated form: first character, length-2, last character.
      cout << s.front() << s.size() - 2 << s.back() << endl;
    }
  }
  return 0; // Indicate successful program termination.
}