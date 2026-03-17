/**
 * @file 71A.CPP
 * @brief Solution to the "Way Too Long Words" problem.
 *
 * @details This program shortens words that are longer than 10 characters. It reads an integer 't'
 * for the number of test cases, then for each case, it reads a word. If the word's length
 * is over 10 characters, it's abbreviated to its first letter, the count of letters
 * in between, and its last letter. Otherwise, the word is printed unchanged.
 *
 * @section
 * Algorithm:
 * 1. Read the number of test cases, t.
 * 2. Loop t times:
 *    a. Read the input string.
 *    b. Check if the string length is greater than 10.
 *    c. If it is, print the abbreviation.
 *    d. If not, print the original string.
 *
 * Time Complexity: O(T * L), where T is the number of test cases and L is the maximum length of a string.
 * Space Complexity: O(L), to store the string for each test case.
 */
// 71A Way Too Long Words
// https://codeforces.com/problemset/problem/71/A

#include <bits/stdc++.h>

using namespace std;

int main() {
  int t;
  cin >> t;
  /**
   * @brief This loop processes each of the 't' test cases.
   * Invariant: The loop will execute exactly 't' times.
   */
  while (t--) {
    string str;
    cin >> str;
    int n = str.length();
    /**
     * @brief This block determines whether to print the word as is or to abbreviate it.
     * Pre-condition: 'str' holds the word and 'n' holds its length.
     */
    if (n <= 10) {
      // If the word's length is 10 or less, print it directly.
      cout << str << '\n';
    } else {
      // If the word is longer than 10 characters, create and print the abbreviation.
      // Format: first character, number of middle characters, last character.
      cout << str[0] << n - 2 << str[n - 1] << '\n';
    }
  }
  return 0;
}