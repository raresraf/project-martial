/*
Author : hardRocK
Date :
Online Judge : Codeforces
Problem Name : Way Too Long Words
Problem Link : http://codeforces.com/problemset/problem/71/A
*/

/**
 * @file 71A.CPP
 * @brief Solves the "Way Too Long Words" problem from Codeforces.
 * @details The task is to process a list of words. If a word is longer than 10
 * characters, it should be abbreviated. Otherwise, it should be printed as is.
 * The abbreviation consists of the first character, the number of characters
 * between the first and last, and the last character.
 *
 * @algorithm
 * 1. Read the number of words to process, T.
 * 2. Loop T times:
 *    a. Read the word into a string `s`.
 *    b. Check the length of the string `s`.
 *    c. If the length is 10 or less, print the string unchanged.
 *    d. If the length is greater than 10, print the first character `s[0]`,
 *       followed by the integer `s.length() - 2`, followed by the last
 *       character `s[s.length() - 1]`.
 *
 * @complexity
 * Time Complexity: O(T * L), where T is the number of words and L is the
 * maximum length of a word. This is because for each word, we perform
 * operations proportional to its length (reading it).
 * Space Complexity: O(L) to store the longest word.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {

  int T; // Number of test cases (words).
  cin >> T;

  /**
   * @block
   * @brief Process each word one by one.
   */
  while (T--) {

    string s;
    cin >> s;

    /**
     * @brief Check if the word needs to be abbreviated.
     */
    if (s.size() <= 10) {
      // If the word is not "too long", print it as is.
      cout << s << endl;
    } else {
      // Otherwise, print the abbreviation.
      // s[0]: First character.
      // s.size() - 2: Number of characters between the first and last.
      // s[s.size() - 1]: Last character.
      cout << s[0] << s.size() - 2 << s[s.size() - 1] << endl;
    }
  }

  // system("pause");
  return 0;
}
