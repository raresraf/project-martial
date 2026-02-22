/**
 * @file 58A.CPP
 * @brief Solution to Codeforces problem 58A - Chat room.
 *
 * @algorithm
 * The problem is to determine if the word "hello" can be spelled out from a given
 * string by deleting zero or more characters. This is equivalent to checking if
 * "hello" is a subsequence of the input string.
 *
 * This solution uses a greedy two-pointer approach, which is an efficient way
 * to solve subsequence problems.
 *
 * The algorithm is as follows:
 * 1. Define the target string "hello".
 * 2. Use one pointer `i` to iterate through the input string and another pointer
 *    `j` to track the current character being sought in the target "hello".
 * 3. As we iterate through the input string with `i`, if the character `str[i]`
 *    matches the target character `hello[j]`, we advance the target pointer `j`.
 * 4. After iterating through the entire input string, if the pointer `j` has
 *    reached the end of the target string (i.e., `j == 5`), it means all
 *    characters of "hello" were found in the correct order.
 * 5. Print "YES" if the subsequence was found, otherwise "NO".
 *
 * @note This file includes a large number of preprocessor macros for common
 *       operations, a style often seen in competitive programming.
 *
 * @complexity
 * - Time Complexity: O(N), where N is the length of the input string, because
 *   it involves a single pass through the string.
 * - Space Complexity: O(N) to store the input string.
 */
/* Monil Soni sonil06 */
/* DA-IICT  */

#include <bits/stdc++.h>

using namespace std;

typedef long long LL;
typedef vector<int> vi;
typedef vector<long long> vl;

#define MOD 1000000007
#define _for(i, n) for (int i = 0; i < n; i++)
#define __for(i, n) for (int i = 1; i <= n; i++)
#define REP(i, s, n) for (int i = (s), _n = (n); i <= _n; i++)
#define FOR(i, n) for (int i = 0, _n = (n); i < _n; i++)
#define REPD(i, e, s) for (int i = (e), _s = (s); i >= _s; i--)
#define tr(container, it)                                                      \
  for (typeof(container.begin()) it = container.begin();                       \
       it != container.end(); it++)
#define pb push_back
#define mp make_pair
#define fi first
#define se second
#define ALL(x) x.begin(), x.end()

int main() {
  string str;
  string hello = "hello";
  cin >> str;
  int j = 0; // Pointer for the target string "hello"
  int n = str.size();
  // Optimization: If the input string is shorter than "hello", it can't be a subsequence.
  if (n < 5) {
    printf("NO\n");
    return 0;
  }
  /**
   * @brief Iterates through the input string `str` to find "hello" as a subsequence.
   * `i` is the pointer for the input string `str`.
   * `j` is the pointer for the target string `hello`.
   */
  for (int i = 0; i < n; i++) {
    // If the current character in `str` matches the current character in `hello`,
    // advance the pointer for `hello`.
    if (j < 5 && str.at(i) == hello.at(j))
      j++;
  }
  /**
   * @brief Check if the entire "hello" subsequence was found.
   * If `j` is 5, it means all characters of "hello" were matched in order.
   */
  if (j == 5)
    printf("YES\n");
  else
    printf("NO\n");
  return 0;
}
