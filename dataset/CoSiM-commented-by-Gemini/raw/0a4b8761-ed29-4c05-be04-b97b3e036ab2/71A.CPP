/**
 * @file 71A.CPP
 * @brief Solves the "Way Too Long Words" problem.
 * @details This program reads an integer `n` followed by `n` strings. For each string, if its length is greater than 10,
 * it prints a shortened version: the first character, the number of characters between the first and last, and the last character.
 * Otherwise, it prints the original string. This is a common task in competitive programming.
 *
 * @algorithm Abbreviation for long strings.
 * @complexity
 * Time complexity: O(N * L) where N is the number of strings and L is the maximum length of a string, for reading the input. The processing of each string is O(1).
 * Space complexity: O(L) for storing each string.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {

  // n: the number of words to process.
  int n;
  cin >> n;
  /**
   * @brief Main processing loop.
   * @invariant Loops `n` times to process each word.
   */
  while (n--) {
    string s;
    cin >> s;
    /**
     * @brief Conditional abbreviation.
     * @pre A string `s` has been read.
     * @post If the string's length is greater than 10, it's abbreviated. Otherwise, it's printed as is.
     */
    if (s.length() > 10)
      // Print the first character, the length of the middle part, and the last character.
      cout << s[0] << (s.length() - 2) << s[s.length() - 1] << endl;
    else
      // If the string is not "too long", print it as is.
      cout << s << endl;
  }
  return 0;
}