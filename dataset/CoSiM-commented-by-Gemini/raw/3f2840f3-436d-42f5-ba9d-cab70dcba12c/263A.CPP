/**
 * @file 263A.CPP
 * @brief Solution for the "Beautiful Matrix" problem (Codeforces 263A).
 * @author Moshiur Rahman Mohim
 *
 * @algorithm
 * The problem provides a 5x5 matrix containing a single '1' and 24 '0's. The
 * goal is to find the minimum number of moves (swapping adjacent rows or columns)
 * to move the '1' to the center of the matrix. The center is at position (3,3)
 * in a 1-indexed grid.
 *
 * The minimum number of moves is equivalent to the Manhattan distance between the
 * current position of the '1' and the center. The formula for Manhattan distance
 * between (x1, y1) and (x2, y2) is `|x1 - x2| + |y1 - y2|`.
 *
 * The algorithm is:
 * 1. Iterate through the 5x5 grid to find the coordinates (x, y) of the '1'.
 * 2. Calculate the Manhattan distance to the center (3,3): `abs(x - 3) + abs(y - 3)`.
 * 3. Print the result.
 *
 * @complexity
 * - Time Complexity: O(1), as the grid size is fixed at 5x5 (25 operations).
 * - Space Complexity: O(1), for storing the fixed-size grid.
 */

/* ===================================*/
/* Author :   Moshiur Rahman Mohim    */
/* Email  :   mtmohim74@gmail.com     */
/* Institute: HSTU                    */
/* ===================================*/

// Note: The following section contains a large amount of code (includes, macros,
// typedefs, and helper functions) that is part of a generic competitive
// programming template. Most of it is not used for solving this specific problem.
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include <algorithm>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <vector>
using namespace std;

#define pi acos(-1.0)
#define mod 1000000007
#define MAX 10000
#define FOR(i, b, n) for (LL i = b; i <= n; i++)
#define FORR(i, n, b) for (LL i = n; i >= b; i--)
// Library.
#define mem(a, b) memset(a, b, sizeof(a))
#define Sort(x) sort(x.begin(), x.end())
#define Reverse(x) reverse(x.begin(), x.end())
#define gcd(a, b) __gcd(a, b)
#define lcm(a, b) (a * (b / gcd(a, b)))
#define sq(x) (x) * (x)
// File.
#define READ(f) freopen(f, "r", stdin)
#define WRITE(f) freopen(f, "w", stdout)
// i/o.
#define sf scanf
#define pf printf
#define pr1(x) cout << x << "\n"
#define pr2(x, y) cout << x << " " << y << "\n"
#define pr3(x, y, z) cout << x << " " << y << " " << z << "\n"
// Test Case.
#define TEST(i, t)                                                             \
  LL i, t;                                                                     \
  scanf("%I64d", &t);                                                          \
  for (i = 1; i <= t; i++)

typedef unsigned int U;
typedef long int L;
typedef unsigned long int LU;
typedef long long int LL;
typedef unsigned long long int LLU;
typedef float F;
typedef double LF;
typedef char C;

// This function is part of the template and is not used in this solution.
int BigMod(LL B, LL P, LL M) {
  LL R = 1;
  while (P > 0) {
    if (P % 2 == 1)
      R = (R * B) % M;
    P /= 2;
    B = (B * B) % M;
  }
  return (int)R;
}

/*............End............*/
/*.......Global Declaration.......*/
LL ch[10][10]; // Global 2D array to store the matrix.
int main() {
  // READ("input.txt");
  // WRITE("output.txt");
  LL n, x, y; // x and y will store the 1-indexed coordinates of the '1'.
  /**
   * @brief Loop through the 5x5 grid to read inputs and find the '1'.
   * The loops use 1-based indexing, from 1 to 5.
   */
  FOR(i, 1, 5) { // Macro expands to a for loop for rows.
    FOR(j, 1, 5) { // Macro expands to a for loop for columns.
      cin >> ch[i][j];
      if (ch[i][j] == 1) {
        // Store the 1-indexed coordinates of the '1'.
        x = i;
        y = j;
      }
    }
  }
  /**
   * @brief Calculate and print the Manhattan distance to the center (3,3).
   * This is the minimum number of row/column swaps required.
   */
  cout << abs(3 - x) + abs(3 - y);
  return 0;
}
