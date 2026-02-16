/**
 * @file 160A.CPP
 * @author Moshiur Rahman Mohim (mtmohim74@gmail.com)
 * @brief This program solves a competitive programming problem related to "lucky numbers".
 *
 * @description
 * The program reads two integers, `n` and `k`. It then reads `n` additional
 * integers. For each of these `n` integers, it counts the number of "lucky
 * digits" ('4' or '7') it contains. If this count is less than or equal to `k`,
 * a primary counter is incremented. Finally, the program prints the total count
 * of numbers that satisfied this condition.
 *
 * @algorithm
 * 1. Read `n` and `k`.
 * 2. Initialize a counter `s` to 0.
 * 3. Loop `n` times:
 *    a. Read an integer `x`.
 *    b. Initialize a temporary digit counter `c` to 0.
 *    c. Use a nested loop to extract each digit of `x`.
 *    d. If a digit is 4 or 7, increment `c`.
 *    e. After checking all digits of `x`, if `c <= k`, increment `s`.
 * 4. Print the final value of `s`.
 *
 * @time_complexity O(N * D), where N is the number of integers to check, and D
 *   is the maximum number of digits in any of those integers.
 * @space_complexity O(1).
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

const int INF = 0x7f7f7f7f;

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

int main() {
  // READ("input.txt");
  // WRITE("output.txt");
  LL n, k, x, s, c, r;
  // This loop handles multiple test cases if provided in the input stream.
  while (cin >> n >> k) {
    // 's' is the main counter for numbers that satisfy the condition.
    s = 0;
    // Block Logic: Process 'n' integers one by one.
    FOR(i, 1, n) {
      cin >> x;
      // 'c' is the counter for lucky digits ('4' or '7') in the current number 'x'.
      c = 0;
      // Block Logic: Extract and check each digit of the number 'x'.
      while (x != 0) {
        r = x % 10; // Extract the last digit.
        x /= 10;    // Remove the last digit.
        if (r == 4 || r == 7)
          c += 1;
      }
      // Pre-condition: Check if the count of lucky digits is within the allowed limit 'k'.
      if (c <= k)
        s += 1;
    }
    cout << s << endl;
  }
  return 0;
}
