/**
 * @file 110A.CPP
 * @brief Solves the "Nearly Lucky Number" problem.
 * @details A number is considered "nearly lucky" if the count of its "lucky digits" ('4' and '7')
 * is itself a lucky number (i.e., 4 or 7). This program reads a number as a string, counts the
 * occurrences of the digits '4' and '7', and then checks if this count is equal to 4 or 7.
 *
 * @algorithm Digit counting and comparison.
 * @complexity
 * Time complexity: O(L), where L is the number of digits in the input number, because it iterates through the string once.
 * Space complexity: O(L) to store the input number as a string.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  string s;
  // The while loop is a common pattern for multiple test cases, but this problem expects a single input.
  while (cin >> s) {
    // i: loop counter. j: unused. k: counter for lucky digits. l: length of the string.
    int i, j, k, l;
    l = s.length();
    k = 0; // Initialize the lucky digit counter.

    /**
     * @brief Counts the number of lucky digits ('4' or '7').
     * @invariant Iterates through each character of the input string `s`.
     */
    for (i = 0; i < l; i++) {
      if (s[i] == '4' || s[i] == '7') {
        k++;
      }
    }

    /**
     * @brief Checks if the count of lucky digits is itself a lucky number.
     * @post Prints "YES" if the count is 4 or 7, otherwise prints "NO".
     */
    if (k == 4 || k == 7) {
      cout << "YES" << endl;
    } else {
      cout << "NO" << endl;
    }
  }
  return 0;
}