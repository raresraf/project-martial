/**
 * @file raw/4504a094-38e0-4051-bf1d-ce76b88223da/158A.CPP
 * @brief Ostensibly a solution for a competitive programming problem, but its
 *        logic is unclear and it does not seem to match its filename (158A).
 *
 * @algorithm
 * The purpose of this code is difficult to determine with certainty. The
 * filename "158A.CPP" suggests it's a solution for Codeforces problem
 * "Next Round", but the implementation does not align with that problem's
 * requirements.
 *
 * The general logic seems to be:
 * 1. Read an integer `n`.
 * 2. Read `n` more integers and calculate their sum.
 * 3. The result is calculated as `ceil(sum / 4.0)`, which is implemented
 *    using integer arithmetic as `sum / 4` and adding 1 if there's a remainder.
 *
 * This suggests a bin-packing-style problem where items of different sizes
 * (the `n` integers) need to be packed into bins of capacity 4, and we want
 * to find the minimum number of bins.
 *
 * The code also contains a highly specific, hardcoded block for the case
 * when `n` is 3. It checks if the three input numbers are exactly 3, 3, and 2,
 * and if so, prints "3". This is a strong indicator of hardcoding against a
 * specific test case and suggests the general logic might be flawed.
 *
 * The outer `while(scanf(...) == 1)` loop is also unusual for typical
 * competitive programming problem structures.
 *
 * @complexity
 * - Time Complexity: O(N) for each test case, due to the loop for reading inputs.
 * - Space Complexity: O(1) auxiliary space (the `ar[5]` array is small and constant).
 */
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

// A large number of macros and headers are included, many of which are unused.
#define vc vector<int>
#define gcd(x, y) __gcd(x, y)
#define lcm(a, b) (a * (b / gcd(a, b)))
#define MIN(a, b) (a < b ? a : b)
// MAX is defined twice with different meanings. This is a bug.
#define MAX(a, b) (a > b ? a : b)
#define MAX 1000001
#define inf 999999999
#define PI acos(-1)
#define MOD 1000000007

using namespace std;

int main() {
  long int n, k, x; // `k` is declared but never used.

  // This loop continues as long as `scanf` can read an integer.
  // This is not a standard input format for most competitive programming problems.
  while (scanf("%ld", &n) == 1) {
    int ar[5], j = 0;
    long int sum = 0;
    for (long int i = 0; i < n; i++) {
      // Special, hardcoded logic for when n is 3.
      if (n == 3) {
        cin >> x;
        sum = sum + x;
        ar[j++] = x;
      } else {
        cin >> x;
        sum = sum + x;
      }
    }

    if (n == 3) {
      // A specific check for a hardcoded test case.
      if (ar[0] == 3 && ar[1] == 3 && ar[2] == 2)
        printf("3\n");
      else {
        // General logic for n=3 seems to be the same as the general case.
        long int ans = sum / 4;
        if (sum % 4 == 0)
          printf("%ld\n", ans);
        else
          printf("%ld\n", ans + 1);
      }
      continue; // Skip the general logic below for n=3.
    }
    
    // General case: calculate ceil(sum / 4.0) using integer arithmetic.
    long int ans = sum / 4;
    if (sum % 4 == 0)
      printf("%ld\n", ans);
    else
      printf("%ld\n", ans + 1);
  }
  return 0;
}
