/**
 * @file 58A.CPP
 * @brief This program determines if the word "hello" can be formed as a subsequence
 *        from a given input string. A subsequence means the characters of "hello"
 *        appear in the input string in the correct order, but not necessarily contiguously.
 * Algorithm: Subsequence matching (linear scan).
 * Time Complexity: O(L) where L is the length of the input string, as it iterates
 *                  through the string at most once.
 * Space Complexity: O(L) for storing the input string. The target string "hello" is O(1).
 */
#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#define REP(i, n) for (int i = 0; i < (int)(n); ++i)
#define FOR(i, m, n) for (int i = (m); i < (int)(n); ++i)
#define ALL(x) (x).begin(), (x).end()
#define INF 2000000000

#ifdef LOCAL
#define dprintf(...) fprintf(stdout, __VA_ARGS__)
#else
#define dprintf(...) 0
#endif

using namespace std;

typedef long long LL;
typedef unsigned long long ULL;
typedef unsigned int uint;

const double PI = 3.1415926535897932384626433832795;

int main() {
  string s; // The input string to check for the subsequence.
  // Reads the input string.
  cin >> s;
  string hello = "hello"; // The target subsequence string.
  int goodIdx = 0; // Index to track the progress of matching characters in the `hello` string.
  /**
   * Block Logic: Iterates through the input string `s` to find characters of `hello`
   *              in their correct sequential order.
   * Invariant: `goodIdx` represents the number of characters of `hello` that have
   *            been successfully matched as a subsequence so far.
   */
  REP(i, s.size()) { // Iterates through each character of the input string.
    // Checks if the current character of `s` matches the character `hello[goodIdx]`.
    if (s[i] == hello[goodIdx]) {
      goodIdx++; // If matched, advance to the next character in `hello`.
      // If all characters of "hello" have been matched.
      if (goodIdx >= hello.size()) {
        break; // No need to continue scanning the rest of the string `s`.
      }
    }
  }
  /**
   * Block Logic: Determines if the entire "hello" word was found as a subsequence.
   * Invariant: If `goodIdx` equals the length of "hello", it means all characters were found.
   */
  if (goodIdx == hello.size()) { // If `goodIdx` reached the length of "hello".
    cout << "YES" << endl;       // "hello" can be formed as a subsequence.
  } else {
    cout << "NO" << endl;        // "hello" cannot be formed.
  }

  return 0;
}
