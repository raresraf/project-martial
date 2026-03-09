/**
 * @file 281A.CPP
 * @brief Solution to Codeforces Problem 281A: "Word Capitalization".
 *        This program reads a single word and capitalizes its first letter,
 *        leaving the rest of the word unchanged.
 *
 * Algorithm: String manipulation involving direct character access and case conversion.
 * Time Complexity: O(L) where L is the length of the input word, due to reading the string
 *                  and a single character modification.
 * Space Complexity: O(L) to store the input word.
 *
 * Original comments:
 * wordcapitalization.cpp
 * http://codeforces.com/problemset/problem/281/A
 * Created on: Oct 26, 2017
 * Author: athena
 * 31761868 	GNU C++14 	Accepted 	30 ms 	8 KB 	2017-10-26
 * 19:22:42 	2017-10-26 19:22:43
 */

#include <iostream>
#include <string>
#include <cctype> // Required for toupper
using namespace std;

int main() {
  /**
   * Functional Utility: Manages the input, capitalization logic, and output of the word.
   */
  string s; // Declares a string to store the input word.
  cin >> s;  // Reads the word from standard input.

  s[0] = toupper(s[0]); // Converts the first character of the string to uppercase.

  cout << s << endl; // Prints the modified string to standard output, followed by a newline.

  return 0; // Indicate successful program termination.
}
