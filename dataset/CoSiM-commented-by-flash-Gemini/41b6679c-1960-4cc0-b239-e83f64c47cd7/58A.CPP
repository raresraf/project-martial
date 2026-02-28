/**
 * @file 58A.CPP
 * @brief Checks if the string "hello" is a subsequence of a given input string.
 *
 * This program reads an input string and determines if the characters 'h', 'e', 'l', 'l', 'o'
 * appear in that specific order within the input string, not necessarily contiguously.
 *
 * Algorithm: Subsequence matching using a two-pointer approach. One pointer iterates
 *            through the input string, and another tracks the progress of matching
 *            the target subsequence ("hello").
 * Time Complexity: O(L) where L is the length of the input string, as it performs
 *                  a single pass over the string.
 * Space Complexity: O(1) beyond the storage for the input string, as only a few
 *                   scalar variables and the "hello" string are used.
 */

/* Monil Soni sonil06 */
/* DA-IICT  */

#include <bits/stdc++.h>

using namespace std;

typedef long long LL;
typedef vector<int> vi;
typedef vector<long long> vl;

// @brief Modulo constant, commonly used in problems involving large numbers to prevent overflow.
#define MOD 1000000007
// @brief Macro for a for loop from 0 to n-1 (inclusive).
#define _for(i, n) for (int i = 0; i < n; i++)
// @brief Macro for a for loop from 1 to n (inclusive).
#define __for(i, n) for (int i = 1; i <= n; i++)
// @brief Macro for a for loop from s to n (inclusive).
#define REP(i, s, n) for (int i = (s), _n = (n); i <= _n; i++)
// @brief Macro for a for loop from 0 to n-1 (inclusive).
#define FOR(i, n) for (int i = 0, _n = (n); i < _n; i++)
// @brief Macro for a reverse for loop from e down to s (inclusive).
#define REPD(i, e, s) for (int i = (e), _s = (s); i >= _s; i--)
// @brief Macro for iterating through a container (legacy C++).
#define tr(container, it)                                                      \
  for (typeof(container.begin()) it = container.begin();                       \
       it != container.end(); it++)
// @brief Shorthand for vector::push_back.
#define pb push_back
// @brief Shorthand for std::make_pair.
#define mp make_pair
// @brief Shorthand for pair::first.
#define fi first
// @brief Shorthand for pair::second.
#define se second
// @brief Shorthand for getting iterators to the beginning and end of a container.
#define ALL(x) x.begin(), x.end()

int main() {
  /**
   * @brief Main function to determine if "hello" is a subsequence of the input string.
   *
   * Initializes target string "hello", reads an input string, performs an initial
   * length check, and then iterates through the input string to find characters
   * of "hello" in sequential order. Finally, it prints "YES" or "NO" based on the result.
   */
  // @param str Stores the input string from the user.
  string str;
  // @param hello Stores the target subsequence string.
  string hello = "hello";
  // Reads the input string from standard input.
  cin >> str;
  // @param j Pointer/index for the `hello` string, tracking matched characters.
  int j = 0;
  // @param n Stores the length of the input string `str`.
  int n = str.size();
  // @brief Initial check: if the input string is shorter than "hello", it cannot contain "hello" as a subsequence.
  if (n < 5) {
    printf("NO\n");
    return 0;
  }
  /**
   * @brief Loop to iterate through the input string and match characters of "hello" as a subsequence.
   * Invariant: `hello.at(0...j-1)` has been found as a subsequence in `str.at(0...i-1)`.
   */
  for (int i = 0; i < n; i++) {
    /**
     * @brief Conditional logic to advance the `hello` subsequence pointer.
     * Logic: If the current character of the input string `str.at(i)` matches
     *        the next expected character of `hello.at(j)`, and `j` is within
     *        bounds, increment `j`.
     */
    if (j < 5 && str.at(i) == hello.at(j))
      j++;
  }
  /**
   * @brief Final check: If all characters of "hello" were matched in sequence, print "YES", otherwise "NO".
   */
  if (j == 5)
    printf("YES\n");
  else
    printf("NO\n");
  return 0;
}
