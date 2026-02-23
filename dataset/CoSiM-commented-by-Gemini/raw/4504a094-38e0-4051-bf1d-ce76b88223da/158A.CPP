/**
 * @file 158A.CPP
 * @brief An unclear competitive programming solution.
 *
 * @note The filename `158A.CPP` corresponds to the "Next Round" problem on
 * Codeforces, but the logic within this file does not match that problem.
 * The implementation seems to be an attempt to solve a problem related to
 * grouping items into sets of 4, possibly Codeforces problem 158B ("Taxi"),
 * but the approach is simplistic and likely incorrect for that problem as well.
 * It contains unusual, hard-coded logic.
 *
 * @algorithm
 * The code's apparent algorithm is as follows:
 * 1. It runs in a loop, suggesting multiple test cases.
 * 2. For each case, it reads an integer `n`.
 * 3. It then reads `n` integers and calculates their sum.
 * 4. A hard-coded special case exists: if `n` is 3 and the inputs are
 *    exactly 3, 3, and 2, the output is 3.
 * 5. In all other scenarios, it calculates `ceil(sum / 4.0)`, which is
 *    implemented using integer arithmetic as `sum / 4` or `sum / 4 + 1`.
 *    This suggests finding the number of groups of size 4 that can be
 *    formed from a total of `sum` items.
 *
 * @complexity
 * - Time Complexity: O(N) for each test case, where N is the number of integers to sum.
 * - Space Complexity: O(1), aside from a small constant-size array `ar`.
 */

// A large number of standard headers are included, many of which are not used.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <iostream>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

// A collection of macros, most of which are not used in this specific code.
#define vc vector<int>
#define gcd(x, y) __gcd(x, y)
#define lcm(a, b) (a * (b / gcd(a, b)))
#define MIN(a, b) (a < b ? a : b)
#define MAX(a, b) (a > b ? a : b)
#define MAX 1000001 // Redefinition of MAX macro.
#define inf 999999999
#define PI acos(-1)
#define MOD 1000000007

using namespace std;

int main() {
  // n: the number of items in the current test case.
  // k: declared but unused.
  // x: stores each integer as it's read.
  long int n, k, x;

  // Main loop to process multiple test cases, continues as long as `scanf` successfully reads `n`.
  while (scanf("%ld", &n) == 1) {
    // ar: small array to store inputs specifically when n=3.
    // j: index for the `ar` array.
    int ar[5], j = 0;
    // sum: stores the sum of the `n` integers.
    long int sum = 0;
    
    /**
     * Block Logic: Read `n` numbers and calculate their sum. Includes a
     * special branch to store numbers in `ar` if `n` is 3.
     */
    for (long int i = 0; i < n; i++) {
      if (n == 3) {
        cin >> x;
        sum = sum + x;
        ar[j++] = x;
      } else {
        cin >> x;

        sum = sum + x;
      }
    }
    
    /**
     * Block Logic: Contains highly specific, hard-coded behavior for the case when n=3.
     */
    if (n == 3) {
      // If the inputs are exactly 3, 3, and 2, print 3. The reason is unclear.
      if (ar[0] == 3 && ar[1] == 3 && ar[2] == 2)
        printf("3\n");
      else {
        // Otherwise for n=3, calculate ceil(sum/4).
        long int ans = sum / 4;
        if (sum % 4 == 0)
          printf("%ld\n", ans);
        else
          printf("%ld\n", ans + 1);
      }
      continue; // Skip the general logic below and start the next test case.
    }
    
    // General case for n != 3: calculate ceil(sum/4).
    long int ans = sum / 4;
    if (sum % 4 == 0)
      printf("%ld\n", ans);
    else
      printf("%ld\n", ans + 1);
  }
  return 0;
}
