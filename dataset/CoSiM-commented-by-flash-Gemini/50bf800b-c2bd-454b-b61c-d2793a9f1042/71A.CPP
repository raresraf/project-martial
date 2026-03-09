/*
Author : hardRocK
Date :
Online Judge : Codeforces
Problem Name : Way Too Long Words
Problem Link : http://codeforces.com/problemset/problem/71/A
*/

/**
 * @file 71A.CPP
 * @brief Solution to Codeforces Problem 71A: "Way Too Long Words".
 *        This program processes a series of words, abbreviating those
 *        that are excessively long (length > 10) by replacing their
 *        middle characters with a count, while printing shorter words as is.
 *
 * Algorithm: String length check and conditional formatting. For words longer
 *            than 10 characters, it extracts the first character, calculates
 *            the count of intermediate characters (length - 2), and appends
 *            the last character.
 * Time Complexity: O(L) for each word, where L is the length of the word,
 *                  due to string input and basic character access. Total
 *                  time complexity is O(T * L_max) for T test cases.
 * Space Complexity: O(L_max) to store the longest input word.
 */

#include <bits/stdc++.h> // Includes various standard libraries for C++ competitive programming.
using namespace std;

int main() {
  /**
   * Functional Utility: Manages the execution flow for multiple test cases,
   *                     reading input words and applying the abbreviation logic.
   */

  int T; // Declares an integer to store the number of test cases.
  cin >> T; // Reads the number of test cases from standard input.

  /**
   * Block Logic: Processes each test case.
   * Pre-condition: `T` holds the number of test cases.
   * Invariant: `T` decrements with each iteration; the loop continues until all
   *            test cases are processed.
   */
  while (T--) {

    string s; // Declares a string variable to store the current word for the test case.
    cin >> s; // Reads the word for the current test case.

    /**
     * Block Logic: Checks the length of the current word and applies conditional formatting.
     * Pre-condition: `s` contains the word for the current test case.
     * Invariant: Either the original word or its abbreviated form is printed.
     */
    if (s.size() <= 10) {
      cout << s << endl; // Prints the word as is if its length is 10 or less.
    } else {
      // Prints the abbreviated form: first character, length-2, last character.
      cout << s[0] << s.size() - 2 << s[s.size() - 1] << endl;
    }
  }

  // system("pause"); // Commented out: typically used in Windows for pausing console output, not standard practice.
  return 0;          // Indicates successful program termination.
}
