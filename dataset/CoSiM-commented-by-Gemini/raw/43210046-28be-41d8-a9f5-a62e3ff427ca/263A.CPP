/**
 * @file 263A.CPP
 * @brief Solution to Codeforces problem 263A - Beautiful Matrix.
 *
 * @algorithm
 * The problem is to find the minimum number of moves (Manhattan distance) to
 * move a single '1' in a 5x5 matrix to the center.
 *
 * This implementation is highly memory-efficient. It iterates through the grid's
 * cells, reading one number at a time without storing the entire matrix.
 * As soon as the '1' is found, it calculates the Manhattan distance to the
 * center and prints the result.
 *
 * For a 0-indexed 5x5 grid, the center is at coordinates (2, 2). The Manhattan
 * distance from a point (i, j) to the center is `abs(2 - i) + abs(2 - j)`.
 *
 * @note The file includes a large number of headers and preprocessor macros,
 *       likely part of a standard competitive programming template.
 *
 * @complexity
 * - Time Complexity: O(1), as it iterates through a constant 25 cells at most.
 * - Space Complexity: O(1), as it does not store the matrix.
 */
#include <algorithm>
#include <assert.h>
#include <bitset>
#include <ctype.h>
#include <deque>
#include <iostream>
#include <map>
#include <math.h>
#include <queue>
#include <set>
#include <stack>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <time.h>
#include <vector>

using namespace std;

#define sp system("pause")
#define FOR(i, a, b) for (int i = a; i <= b; ++i)
#define FORD(i, a, b) for (int i = a; i >= b; --i)
#define REP(i, n) FOR(i, 0, (int)n - 1)
#define pb(x) push_back(x)
#define mp(a, b) make_pair(a, b)
#define MS0(x) memset(x, 0, sizeof(x))
#define MS1(x) memset(x, 1, sizeof(x))
#define SORT(a, n) sort(begin(a), begin(a) + n)
#define REV(a, n) reverse(begin(a), begin(a) + n)
#define ll long long
#define pii pair<int, int>
#define MOD 1000000007

int a[6][6]; // This array is declared but not used in the final logic.

int main() {

  int num;
  /**
   * @brief Iterates through a 5x5 grid using 0-based indexing.
   * The code reads each number and processes it immediately without
   * storing the whole matrix.
   */
  REP(i, 5) {
    REP(j, 5) {
      cin >> num;
      /**
       * @brief When the '1' is found, calculate the Manhattan distance and exit.
       * The logic immediately computes the distance to the center (2,2)
       * and prints the result, solving the problem in a single pass.
       */
      if (num == 1) {
        cout << abs(2 - i) + abs(2 - j)
             << endl; // cause indexing is from 0 to n .so middle element will
                      // be (2,2).
      }
    }
  }
  return 0;
}
