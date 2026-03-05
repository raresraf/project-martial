/*
 * waytoolongwords.cpp
 * http://codeforces.com/problemset/problem/71/A
 *  Created on: Oct 21, 2017
 *     Author: athena
 * 31582116 	GNU C++14 	Accepted 	30 ms 	1876 KB
 * 2017-10-21 20:24:32 	2017-10-21 20:24:32
 */

/**
 * @file 71A.CPP
 * @brief Solution for Codeforces Problem 71A, "Way Too Long Words".
 * @details This program shortens words that have a length strictly greater
 * than 10 by replacing their middle characters with a number.
 *
 * Algorithm:
 * 1. Read the number of words, `n`.
 * 2. Loop `n` times to process each word.
 * 3. In each iteration, read a word.
 * 4. Check if the word's length is greater than 10.
 * 5. If it is, print an abbreviation consisting of the first character,
 *    the count of the characters between the first and last (i.e., length - 2),
 *    and the last character.
 * 6. If the length is 10 or less, print the word unchanged.
 *
 * Time Complexity: O(N * L), where N is the number of words and L is the
 * maximum length of a word, due to string input operations.
 * Space Complexity: O(L) for storing each word.
 */
#include <iostream>
using namespace std;

int main() {
  int n;
  cin >> n;

  string str;

  /**
   * @brief This loop processes each of the `n` words.
   */
  for (int i = 0; i < n; i++) {
    cin >> str;

    /**
     * @brief This block implements the core logic of the problem. It checks
     * the length of the string and either prints it as is or prints its
     * abbreviated form.
     */
    if (str.size() <= 10) {
      // If the word's length is 10 or less, print it unchanged.
      cout << str << endl;
    } else {
      // If the word's length is greater than 10, print the abbreviation.
      cout << str[0] << str.size() - 2 << str[str.size() - 1] << endl;
    }
  }

  return 0;
}
