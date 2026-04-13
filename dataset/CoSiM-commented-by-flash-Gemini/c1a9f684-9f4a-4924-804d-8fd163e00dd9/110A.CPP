/**
 * @file 110A.CPP
 * @brief This program determines if a given number (provided as a string)
 *        is a "nearly lucky" number. A number is nearly lucky if the count
 *        of its 'lucky' digits (4s and 7s) is itself a lucky number (i.e., 4 or 7).
 * Algorithm: Digit counting, Conditional check.
 * Time Complexity: O(L) where L is the length of the input number string,
 *                  as it iterates through the string once.
 * Space Complexity: O(L) for storing the input string.
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
  string s; // Input number as a string.
  // Reads the number as a string.
  cin >> s;
  char c; // Unused variable.
  int count = 0; // Counter for lucky digits (4s and 7s).
  /**
   * Block Logic: Iterates through each character (digit) of the input string
   *              and increments 'count' if the digit is '4' or '7'.
   * Invariant: 'count' stores the total occurrences of lucky digits in the number.
   */
  REP(i, s.size()) { // Loop through each character of the string.
    if (s[i] == '4' || s[i] == '7') { // Checks if the current digit is a '4' or '7'.
      count++; // Increments the lucky digit counter.
    }
  }
  /**
   * Block Logic: Checks if the count of lucky digits itself is a lucky number (4 or 7).
   * Invariant: Prints "YES" if 'count' is 4 or 7, otherwise prints "NO".
   */
  if (count == 4 || count == 7) { // Determines if the count of lucky digits is a lucky number.
    cout << "YES" << endl; // The number is nearly lucky.
  } else {
    cout << "NO" << endl; // The number is not nearly lucky.
  }
  return 0;
}
