/**
 * @file 263A.CPP
 * @brief Solution for competitive programming problem: "Beautiful Matrix".
 *
 * @details This program finds the minimum number of moves required to bring the
 *          number '1' to the center of a 5x5 matrix. A move consists of
 *          shifting the '1' one cell up, down, left, or right. The solution is
 *          calculated as the Manhattan distance from the cell containing '1'
 *          to the center cell (3,3) in a 1-based indexed matrix.
 *          Algorithm: Manhattan Distance calculation.
 *          Time Complexity: O(1) as it involves a fixed-size loop (5x5 matrix)
 *                           and constant-time arithmetic operations.
 *          Domain: Competitive Programming.
 */
/* ===================================*/
/* Author :   Moshiur Rahman Mohim    */
/* Email  :   mtmohim74@gmail.com     */
/* Institute: HSTU                    */
/* ===================================*/

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

// Macro for mathematical constant pi.
#define pi acos(-1.0)
// Modulo constant, commonly used in problems involving large numbers.
#define mod 1000000007
// Maximum value constant, often used for array sizes or limits.
#define MAX 10000
// Macro for a FOR loop (1-based indexing for LL type).
#define FOR(i, b, n) for (LL i = b; i <= n; i++)
// Macro for a reverse FOR loop (1-based indexing for LL type).
#define FORR(i, n, b) for (LL i = n; i >= b; i--)
// Library.
// Macro for memory initialization (sets a block of memory to a specified value).
#define mem(a, b) memset(a, b, sizeof(a))
// Macro for sorting a container.
#define Sort(x) sort(x.begin(), x.end())
// Macro for reversing a container.
#define Reverse(x) reverse(x.begin(), x.end())
// Macro for greatest common divisor.
#define gcd(a, b) __gcd(a, b)
// Macro for least common multiple.
#define lcm(a, b) (a * (b / gcd(a, b)))
// Macro for squaring a value.
#define sq(x) (x) * (x)
// File.
// Macro for redirecting standard input from a file.
#define READ(f) freopen(f, "r", stdin)
// Macro for redirecting standard output to a file.
#define WRITE(f) freopen(f, "w", stdout)
// i/o.
// Macro for formatted input (scanf).
#define sf scanf
// Macro for formatted output (printf).
#define pf printf
// Macro for printing a single variable followed by a newline.
#define pr1(x) cout << x << "\n"
// Macro for printing two variables followed by a newline.
#define pr2(x, y) cout << x << " " << y << "\n"
// Macro for printing three variables followed by a newline.
#define pr3(x, y, z) cout << x << " " << y << " " << z << "\n"
// Test Case.
// Macro for competitive programming test case loop structure.
#define TEST(i, t)                                                             \
  LL i, t;                                                                     \
  scanf("%I64d", &t);                                                          \
  for (i = 1; i <= t; i++)

typedef unsigned int U;             // Unsigned integer.
typedef long int L;                 // Long integer.
typedef unsigned long int LU;        // Unsigned long integer.
typedef long long int LL;           // Long long integer (64-bit).
typedef unsigned long long int LLU;  // Unsigned long long integer.
typedef float F;                    // Single-precision floating-point.
typedef double LF;                  // Double-precision floating-point.
typedef char C;                     // Character type.

// Compute b^p%m
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
// 2D array representing the 5x5 matrix.
LL ch[10][10];
int main() {
  // READ("input.txt"); // Commented out: file input/output redirection.
  // WRITE("output.txt"); // Commented out: file input/output redirection.
  LL n, x, y; // Declare variables: 'n' (unused), 'x', 'y' for coordinates of '1'.
  // Block Logic: Read the 5x5 matrix and find the coordinates of the '1'.
  // Invariant: After this nested loop, 'x' and 'y' will store the 1-based row and column of '1'.
  FOR(i, 1, 5) { // Outer loop for rows (1 to 5).
    FOR(j, 1, 5) { // Inner loop for columns (1 to 5).
      cin >> ch[i][j]; // Read the cell value.
      if (ch[i][j] == 1) { // If the value is '1'.
        x = i;           // Store its row.
        y = j;           // Store its column.
      }
    }
  }
  // Calculate and print the Manhattan distance from (x,y) to the center (3,3).
  cout << abs(3 - x) + abs(3 - y);
  return 0; // Indicate successful program termination.
}
