/**
 * @file 71A.CPP
 * @brief Solves the "Way Too Long Words" problem.
 *
 * This program reads a number `n`, and then processes `n` strings. For each string,
 * if its length is strictly greater than 10, it is abbreviated. The abbreviation
 * consists of the first character of the string, followed by the number of
 * characters between the first and last characters, and finally the last character.
 * If the string's length is 10 or less, it is printed unchanged.
 *
 * @algorithm
 * 1. Read the number of test cases, `n`.
 * 2. Loop `n` times.
 *    a. Read the input string, `s`.
 *    b. Get the length of the string.
 *    c. If `length > 10`:
 *       i. Print the first character (`s[0]`).
 *       ii. Print the number of middle characters (`length - 2`).
 *       iii. Print the last character (`s[length - 1]`).
 *    d. Else (if `length <= 10`):
 *       i. Print the string `s` as is.
 *
 * @complexity
 * Time Complexity: O(n * L), where `n` is the number of strings and `L` is the
 * maximum length of a string. This is dominated by the I/O operations.
 * Space Complexity: O(L) to store the longest string.
 */

#include <bits/stdc++.h>

// Macro for long long int, unused in this specific problem.
#define ll long long int

// Macro for a modulo constant, unused in this specific problem.
#define mod 1000000007

using namespace std;

int main()

{
  // `n` is the number of words to process.
  int n;

  cin >> n;

  /**
   * @brief Loop to process each word.
   */
  while (n--)

  {
    string s;

    cin >> s;

    // Check if the word's length is greater than 10.
    if (s.length() > 10)

    {
      // If it is, print the abbreviation.
      // s[0] is the first character.
      // s.length() - 2 is the count of characters between the first and last.
      // s[s.length() - 1] is the last character.
      cout << s[0] << s.length() - 2 << s[s.length() - 1] << endl;

    }

    else
      // Otherwise, print the word as is.
      cout << s << endl;
  }
}
