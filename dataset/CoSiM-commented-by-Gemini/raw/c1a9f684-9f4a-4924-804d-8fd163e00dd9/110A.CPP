/**
 * @file 110A.CPP
 * @brief Solution for Codeforces Problem 110A, "Nearly Lucky Number".
 *
 * @details
 * A "lucky number" is a positive integer whose decimal representation contains only the digits 4 and 7.
 * A "nearly lucky number" is a number for which the count of lucky digits (4s and 7s)
 * in its decimal representation is a lucky number.
 *
 * Algorithm:
 * 1. Read the input number as a string.
 * 2. Iterate through the digits of the string and count how many are '4' or '7'.
 * 3. Check if the resulting count is itself a lucky number (i.e., is it 4 or 7?).
 * 4. Print "YES" if the count is 4 or 7, and "NO" otherwise.
 *
 * Time Complexity: O(L), where L is the number of digits in the input number (the length of the string).
 * The algorithm involves a single pass over the string.
 *
 * Space Complexity: O(L), to store the input number as a string.
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

// Macro for a common loop structure (0 to n-1).
#define REP(i, n) for (int i = 0; i < (int)(n); ++i)
// Macro for a flexible loop structure (m to n-1).
#define FOR(i, m, n) for (int i = (m); i < (int)(n); ++i)
// Macro to get iterators to the beginning and end of a container.
#define ALL(x) (x).begin(), (x).end()
// A large integer value often used as infinity.
#define INF 2000000000

// Conditional debug print macro.
#ifdef LOCAL
#define dprintf(...) fprintf(stdout, __VA_ARGS__)
#else
#define dprintf(...) 0
#endif

using namespace std;

// Typedefs for common data types.
typedef long long LL;
typedef unsigned long long ULL;
typedef unsigned int uint;

const double PI = 3.1415926535897932384626433832795;

int main() {
  string s;
  cin >> s;
  int count = 0;
  
  /**
   * @brief This loop counts the number of lucky digits ('4' or '7') in the input string.
   * Invariant: 'count' holds the number of lucky digits found so far.
   */
  REP(i, s.size()) {
    if (s[i] == '4' || s[i] == '7') {
      count++;
    }
  }
  
  /**
   * @brief Check if the count of lucky digits is itself a lucky number (4 or 7).
   * This is the condition for the original number to be "nearly lucky".
   */
  if (count == 4 || count == 7) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }
  
  return 0;
}
