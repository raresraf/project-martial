/**
 * @file 112A.CPP
 * @brief Solution to a competitive programming problem, likely Codeforces 112A "Petya and Strings".
 *
 * @details
 * This program performs a case-insensitive lexicographical comparison of two input
 * strings. It prints -1 if the first string is lexicographically smaller than the
 * second, 1 if it is larger, and 0 if they are equal.
 *
 * @algorithm
 * 1. Read two strings from standard input.
 * 2. Normalize both strings by converting all their characters to lowercase.
 * 3. Use the standard `string::compare` method to perform the comparison on the
 *    normalized strings.
 * 4. Translate the result of the `compare` method into the required output
 *    format (-1, 1, or 0).
 *
 * @complexity
 * - Time Complexity: O(N), where N is the length of the strings. This is determined
 *   by the linear scan for case conversion and the string comparison itself.
 * - Space Complexity: O(N) to store the two input strings.
 */

#include <bits/stdc++.h>

// Macro for a standard for-loop.
#define FOR(i, a, n) for (int i = (int)(a); i < (int)(n); ++i)
// Macro for a for-loop that starts from 0.
#define REP(i, n) FOR(i, 0, n)
#define all(a) a.begin(), a.end()
#define pb push_back

typedef unsigned long long llu;
typedef long long ll;
typedef long double ld;

using namespace std;

/**
 * @brief Main function and entry point of the program.
 */
int main() {
  string a, b;
  // Read the two strings to be compared.
  getline(cin, a);
  getline(cin, b);

  /**
   * @brief Case normalization.
   * To perform a case-insensitive comparison, both strings are converted
   * entirely to lowercase before comparison.
   */
  REP(i, a.size()) {
    a[i] = tolower(a[i]);
    b[i] = tolower(b[i]);
  }

  /**
   * @brief Lexicographical comparison and output.
   * The `string::compare` method returns:
   *  - A negative value if `a` is less than `b`.
   *  - A positive value if `a` is greater than `b`.
   *  - Zero if `a` is equal to `b`.
   * This result is then used to print the required -1, 1, or 0.
   */
  if (a.compare(b))
    printf("%s", a.compare(b) < 0 ? "-1
" : "1
");
  else
    printf("0
");
  return 0;
}
