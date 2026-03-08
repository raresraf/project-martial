/**
 * @file 71A.CPP
 * @brief Solution for the "Way Too Long Words" problem (e.g., Codeforces 71A).
 * @details The program reads a number of words and abbreviates any word that is
 * strictly longer than 10 characters. The abbreviation consists of the first
 * character, the count of the middle characters, and the last character.
 *
 * @algorithm Iterative String Manipulation
 * @complexity
 * - Time: O(N * L), where N is the number of words and L is their maximum length.
 *   The complexity is dominated by the string input operations.
 * - Space: O(L) to store the longest word read in a single iteration.
 */

// --- Includes and Macros ---
// NOTE: This file includes a large number of standard library headers and defines
// many macros that are not used in this specific program. This is a common but
// often unnecessary practice in competitive programming.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <iostream>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#define vc vector<int>
#define gcd(x, y) __gcd(x, y)
#define lcm(a, b) (a * (b / gcd(a, b)))
#define MIN(a, b) (a < b ? a : b)
#define MAX(a, b) (a > b ? a : b)
#define MAX 1000001
#define inf 999999999
#define PI acos(-1)
#define MOD 1000000007

using namespace std;

int main() {
  // tc: test cases, i.e., the number of words to process.
  int tc;
  string s;
  cin >> tc;

  /**
   * @block Word Processing Loop
   * @logic Iterates `tc` times, processing one word in each iteration.
   */
  while (tc--) {
    cin >> s;
    int len = s.size();
    if (len <= 10) {
      // If the word is not longer than 10 characters, print it as is.
      cout << s << endl;
      continue;
    } else {
      // Otherwise, print the abbreviation using C-style printf.
      // Format: [first char][length-2][last char]
      printf("%c%d%c
", s[0], len - 2, s[len - 1]);
    }
  }
  return 0;
}
