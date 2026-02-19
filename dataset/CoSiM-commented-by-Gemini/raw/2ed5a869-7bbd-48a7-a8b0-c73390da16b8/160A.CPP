/**
 * @file 160A.CPP
 * @brief Solution to the "Twins" problem on Codeforces.
 * @details The program determines the minimum number of coins one needs to take from a set
 * so that the sum of their values is strictly greater than the sum of the remaining coins.
 *
 * Author :   Moshiur Rahman Mohim
 * Email  :   mtmohim74@gmail.com
 * Institute: HSTU
 *
 * Algorithm: Greedy Approach
 * The optimal strategy to reach a sum greater than half of the total with the minimum number of coins
 * is to always pick the coins with the highest values first.
 *
 * 1. Read the number of coins, `n`.
 * 2. Read the values of all `n` coins and calculate their total sum, `s`.
 * 3. Sort the coins in non-decreasing (ascending) order.
 * 4. Iterate through the sorted coins in reverse (from largest to smallest).
 * 5. In each step, take the current (largest) coin, add its value to your personal sum `m`,
 *    and increment the count of coins taken, `j`.
 * 6. After taking each coin, check if your current sum `m` is strictly greater than the sum of the
 *    remaining coins, which is `s - m`.
 * 7. As soon as this condition is met, you have found the minimum number of coins required. Print
 *    the count `j` and terminate.
 *
 * Time Complexity: O(N log N), dominated by the sorting operation.
 * Space Complexity: O(N), to store the values of the coins in an array.
 */

/* ===================================*/
/* Author :   Moshiur Rahman Mohim    */
/* Email  :   mtmohim74@gmail.com     */
/* Institute: HSTU                    */
/* ===================================*/

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <vector>
using namespace std;

// The following is a large template of macros and functions common in competitive programming,
// many of which are unused in this specific solution.
#define pi acos(-1.0)
#define mod 1000000007
#define MAX 10000
#define FOR(i, b, n) for (LL i = b; i <= n; i++)
#define FORR(i, n, b) for (LL i = n; i >= b; i--)
#define mem(a, b) memset(a, b, sizeof(a))
#define Sort(x) sort(x.begin(), x.end())
#define Reverse(x) reverse(x.begin(), x.end())
#define gcd(a, b) __gcd(a, b)
#define lcm(a, b) (a * (b / gcd(a, b)))
#define sq(x) (x) * (x)
#define READ(f) freopen(f, "r", stdin)
#define WRITE(f) freopen(f, "w", stdout)
#define sf scanf
#define pf printf
#define pr1(x) cout << x << "\n"
#define pr2(x, y) cout << x << " " << y << "\n"
#define pr3(x, y, z) cout << x << " " << y << " " << z << "\n"
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

// Unused function for modular exponentiation.
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
  LL ch[110], n, s, m, j;
  // The while loop is a common competitive programming pattern, but this problem expects a single test case.
  while (cin >> n) {
    s = 0; // Total sum of all coins.
    
    // Read all coin values and compute the total sum.
    FOR(i, 0, n - 1) {
      sf("%I64d", &ch[i]);
      s += ch[i];
    }
    
    // Sort coins in ascending order to easily access the largest ones.
    sort(ch, ch + n);
    
    m = 0; // Sum of coins taken.
    j = 0; // Number of coins taken.
    
    /**
     * Block Logic: Greedily take the largest coins until their sum exceeds half of the total.
     * Pre-condition: `ch` is sorted, `s` is the total sum.
     * Invariant: `m` is the sum of the `j` largest coins.
     */
    FORR(i, n - 1, 0) {
      j += 1;
      m += ch[i];
      // Check if the sum of coins taken (`m`) is strictly greater than the sum of remaining coins (`s-m`).
      if (m > (s - m)) {
        pr1(j); // Print the minimum number of coins needed.
        break;  // Exit as we have found the solution.
      }
    }
  }
  return 0;
}
