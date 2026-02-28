/**
 * @file 71A.CPP
 * @brief Implements a string abbreviation algorithm.
 *
 * This program reads an integer N, followed by N strings.
 * For each string, if its length exceeds 10 characters, it is
 * abbreviated by printing its first character, followed by its
 * length minus 2, and then its last character. Otherwise, the
 * string is printed as is.
 *
 * Algorithm: Iterative string processing with conditional abbreviation.
 * Time Complexity: O(N * L_max) where N is the number of strings and L_max is the maximum length of a string.
 * Space Complexity: O(L_max) to store the current string.
 */

#include <bits/stdc++.h>
#include <iostream>
#include <math.h>
using namespace std;

int main() {
  /**
   * @brief Main function to process string abbreviations.
   *
   * Initializes variables, reads the number of strings to process,
   * and then iteratively processes each string according to the
   * abbreviation logic.
   */
  int n, temp;
  cin >> n;
  temp = n;
  /**
   * @brief Processes N input strings.
   * Invariant: In each iteration, 'n' decreases, processing one string at a time
   * until all 'temp' strings have been processed.
   */
  while (n--) {
    string a;
    cin >> a;
    int l = a.length();
    /**
     * @brief Conditional logic for string abbreviation.
     *
     * If the string length is greater than 10, it's abbreviated.
     * Otherwise, the original string is printed.
     */
    if (l > 10) {
      int lc = l - 2;
      cout << a.at(0) << lc << a.at(l - 1) << endl;
    } else {
      cout << a << endl;
    }
  }
  return 0;
}
