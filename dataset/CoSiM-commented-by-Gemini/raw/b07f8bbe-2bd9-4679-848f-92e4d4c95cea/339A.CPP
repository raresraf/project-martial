/*
        SubmissionId	:	4353166
        ContestId	:      339
        Index	:      A
        ProblemName	:      Helpful Maths
        ProblemTags	:      ['greedy', 'implementation', 'sortings',
   'strings'] ProgrammingLanguage	:      GNU C++ Verdict	:      OK
*/

/**
 * @file 339A.CPP
 * @brief An efficient C++ solution for sorting digits in a string sum.
 * @details This program parses a string representing a sum (e.g., "1+3+2"),
 * extracts the numbers, sorts them, and then reconstructs the string with the numbers
 * in non-decreasing order. It uses the C++ Standard Library for a clean and efficient implementation.
 *
 * @algorithm
 * 1. Reads the input character by character until the end of the stream.
 * 2. If a character is a digit, it's converted to an integer and stored in a std::vector.
 *    The '+' characters are ignored during parsing.
 * 3. The `std::sort` algorithm is used to sort the vector of integers efficiently.
 * 4. The sorted numbers are then printed, interleaved with '+' characters to form the final string.
 *
 * @complexity
 * - Time: O(D log D), where D is the number of digits in the input. The complexity is dominated
 *   by the `std::sort` operation.
 * - Space: O(D) to store the digits in the vector.
 */

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <vector>

using namespace std;

// Macros for competitive programming, providing shortcuts for common operations.
typedef long long ll;
typedef pair<int, int> pii;
#define X first
#define Y second
#define pb push_back
#define mp make_pair
#define SZ(x) ((int)((x).size()))
#define ALL(x) (x).begin(), (x).end()
#define debug(x) cerr << #x << " = " << x << endl
#define FOREACH(i, c)                                                          \
  for (__typeof((c).begin()) i = (c).begin(); i != (c).end(); i++)
#define FOR(i, a, n) for (__typeof(n) i = (a); i <= (n); i++)
#define FORD(i, n, a) for (__typeof(n) i = (n); i >= (a); i--)
#define MAXN 100010

vector<int> v; // A dynamic array to store the integer operands from the input string.
char c;        // A character variable to read the input stream one character at a time.

int main() {
  // Optimizes I/O by decoupling C++ streams from C standard streams.
  ios::sync_with_stdio(false);
  
  /**
   * @brief Input parsing loop.
   * Reads the expression character by character. Digits are converted to integers and
   * stored in vector `v`, while '+' operators are discarded.
   */
  while (cin >> c) {
    if (c != '+')
      // Converts character '0'-'9' to integer 0-9 by subtracting the ASCII value of '0' (48).
      v.pb(c - 48);
  }

  // Sorts the vector of numbers in non-decreasing order using a standard, efficient sorting algorithm.
  sort(ALL(v));
  
  // Prints the first, smallest number.
  cout << v[0];

  /**
   * @brief Output loop for reconstructing the sum.
   * Iterates through the rest of the sorted vector and prints each number,
   * prefixed by a '+' sign.
   */
  FOR(i, 1, SZ(v) - 1)
  cout << "+" << v[i];
  
  return 0;
}