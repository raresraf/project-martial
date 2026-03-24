/**
 * @file 58A.CPP
 * @brief Solution for Codeforces Problem 58A - "Chat room".
 *
 * @details
 * This program determines if the word "hello" can be formed as a subsequence
 * from a given input string. A subsequence can be formed by deleting zero or
 * more characters from the string without changing the order of the remaining
 * characters.
 *
 * Algorithm:
 * A greedy approach is used. The code iterates through the input string while
 * trying to match characters of the target string "hello" in order.
 * 1. A pointer `goodIdx` is used to track the current character to be found in "hello" (initially 'h').
 * 2. The code iterates through the input string `s`.
 * 3. If the character `s[i]` matches the target character `hello[goodIdx]`, the pointer `goodIdx` is advanced.
 * 4. This continues until either the end of the input string is reached or all
 *    characters of "hello" have been found.
 * 5. If `goodIdx` reaches the length of "hello", it means the subsequence was successfully found.
 *
 * Time Complexity: O(N), where N is the length of the input string, due to a single linear scan.
 * Space Complexity: O(1), as it only uses a few variables regardless of input size.
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

// Common competitive programming macros.
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
  string s;
  cin >> s;
  string hello = "hello";
  
  // `goodIdx` acts as a pointer to the current character we are looking for in the `hello` string.
  int goodIdx = 0;
  
  /**
   * Block Logic: This loop iterates through the input string `s` to find the
   * "hello" subsequence in a greedy manner.
   */
  REP(i, s.size()) {
    // Invariant: If the current character in `s` matches the character we're
    // looking for in `hello`, we advance our position in `hello`.
    if (s[i] == hello[goodIdx]) {
      goodIdx++;
      // Optimization: If we have found all characters of "hello", we can stop early.
      if (goodIdx >= hello.size()) {
        break;
      }
    }
  }

  // Pre-condition: If `goodIdx` equals the length of "hello", it means every
  // character was found in the correct order.
  if (goodIdx == hello.size()) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }

  return 0;
}
