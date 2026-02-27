/**
 * @file 71A.CPP
 * @brief Solves the "Way Too Long Words" problem from Codeforces.
 * @details The program takes a list of words and shortens any word that is
 * longer than 10 characters. The abbreviation consists of the first letter,
 * the number of letters in between, and the last letter.
 *
 * @algorithm
 * 1. Read the number of test cases (words).
 * 2. Loop for each test case:
 *    a. Read the word.
 *    b. If the word's length is greater than 10, print the abbreviation.
 *    c. Otherwise, print the word as is.
 *
 * @complexity
 * Time Complexity: O(T * L), where T is the number of test cases and L is the
 * maximum length of a word, due to I/O operations.
 * Space Complexity: O(L) to store the longest word.
 */
#include <bits/stdc++.h>
using namespace std;

int main(void) {
  int testCases;
  string tooLong; // To store each word.

  cin >> testCases;

  /**
   * @block
   * @brief Process each word provided in the input.
   */
  while (testCases--) {
    cin >> tooLong;

    if (tooLong.size() > 10) {
      // If the word is too long, print its abbreviated form.
      // It uses iterators to get the first and last characters.
      cout << *(tooLong.begin())      // First character
           << (tooLong.length() - 2)  // Number of middle characters
           << *(tooLong.end() - 1)    // Last character
           << endl;
    } else {
      // Otherwise, print the word as is.
      cout << tooLong << endl;
    }
  }

  return 0;
}
