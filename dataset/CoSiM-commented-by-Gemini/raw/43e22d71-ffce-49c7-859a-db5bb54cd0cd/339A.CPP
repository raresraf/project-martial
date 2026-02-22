/**
 * @file 339A.CPP
 * @brief Solution attempt for Codeforces problem 339A - Helpful Maths.
 *
 * @algorithm
 * This code attempts to solve the problem by reading a sequence of numbers from a
 * single line, sorting them, and then printing them separated by '+'.
 *
 * The intended algorithm is:
 * 1. Read numbers from the input line one by one, along with the separator
 *    character, until a newline is found. Store the numbers in an array.
 * 2. Sort the array of numbers.
 * 3. Print the sorted numbers, placing a '+' between each.
 *
 * @warning The input parsing logic is incorrect for the specified problem format.
 *          The problem gives a string like "1+3+2". This code uses
 *          `scanf("%I64d%c", ...)` which is designed to read integers and
 *          characters, not to parse numbers out of a continuous string with
 *          operators. While it might work by coincidence for single-digit inputs,
 *          it is not a robust or correct way to parse the given string format.
 *          It is effectively solving a different problem.
 *
 * @note This file contains a large, mostly unused template of macros and helper
 *       functions common in competitive programming.
 *
 * @complexity
 * - Time Complexity: O(N log N), where N is the number of summands. The complexity
 *   is dominated by the `sort` call.
 * - Space Complexity: O(N), to store the numbers in the `ch` array.
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
C c;
int main() {
  // READ("input.txt");
  // WRITE("output.txt");
  LL ch[100]; // Array to store the numbers read from input.
  LL n;       // Will store the count of numbers.

  /**
   * @brief Reads the input line, attempting to parse numbers and separators.
   * This loop reads a number and a character, stopping when the character is a newline.
   * This parsing method is not robust for the problem's string format.
   */
  FOR(i, 0, 100) {
    sf("%I64d%c", &ch[i], &c);
    if (c == '\n') {
      n = i + 1;
      break;
    }
  }
  // Sort the array of numbers.
  sort(ch, ch + n);
  /**
   * @brief Prints the sorted numbers, adding a '+' as a separator.
   */
  FOR(i, 0, n - 1) {
    pf("%I64d", ch[i]);
    if (i != n - 1)
      pf("+");
  }
  return 0;
}
