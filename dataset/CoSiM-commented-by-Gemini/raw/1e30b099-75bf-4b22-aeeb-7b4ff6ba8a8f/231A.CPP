/**
 * @file 231A.CPP
 * @brief Solution for Codeforces problem 231A, "Team", using a common
 *        competitive programming template.
 *
 * @author Moshiur Rahman Mohim (mtmohim74@gmail.com)
 *
 * @description
 * This program calculates the number of problems a team of three will implement.
 * A problem is implemented only if at least two of the three members are sure
 * about the solution (represented by 1). The program reads the number of
 * problems and then the opinions for each, counting how many meet the criterion.
 *
 * @algorithm
 * The core logic is wrapped in a `TEST` macro which reads the number of test
 * cases (problems) and creates a loop.
 * 1. Initialize a counter `s` to 0.
 * 2. The `TEST` macro reads `n` (as `t`) and loops `n` times.
 * 3. In each iteration, read the three members' opinions (0 or 1).
 * 4. Sum the opinions. If the sum is >= 2, increment the counter `s`.
 * 5. After the loop, print the final count `s`.
 *
 * @time_complexity O(N), where N is the number of problems.
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
/**
 * @def TEST(i, t)
 * @brief A macro to handle a standard competitive programming test case loop.
 * It declares loop variables i and t, reads a 64-bit integer into t, and
 * loops from 1 to t. In this problem, 't' represents the number of problems.
 */
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
  // s: counter for problems that will be solved.
  // a, b, c: opinions of the three members.
  LL s = 0, a, b, c;

  // The TEST macro expands to a loop that runs 't' times, where 't' is the
  // first number read from input (the number of problems).
  TEST(i, t) {
    cin >> a >> b >> c;
    // If the sum of opinions is 2 or 3, the team solves the problem.
    if (a + b + c >= 2)
      s += 1;
  }
  cout << s;
  return 0;
}
