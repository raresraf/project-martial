/**
 * @file 753A.CPP
 * @brief Solution to the "Santa Claus and Candies" problem.
 * @details This program determines the maximum number of children Santa can give a distinct number of candies to,
 * given a total of `n` candies. It also prints the number of candies each child receives.
 * This implementation uses dynamic programming to solve the partition problem.
 *
 * Algorithm: Dynamic Programming (0/1 Knapsack variation)
 * The problem is to find a partition of `n` into the maximum number of distinct positive integers.
 * The code pre-computes the solution for all possible values of `n` up to a limit `N`.
 *
 * - `dp[j]`: Stores the maximum number of distinct integers that sum up to `j`.
 * - `mp[j]`: Stores the previous sum before adding the last number to reach `j`. This acts as a parent pointer for backtracking.
 * - `flag[j]`: A boolean indicating whether a sum `j` is achievable.
 *
 * The DP logic works as follows:
 * 1. Initialize `flag[0] = true` because a sum of 0 is achievable with zero numbers.
 * 2. The outer loop `for (int i = 1; ...)` iterates through the distinct integers `i` (1, 2, 3, ...) that can be used in the sum.
 * 3. The inner loop `for (int j = N - 1; ...)` iterates downwards from the maximum possible sum. This is crucial to ensure that each number `i` is used at most once for a given sum `j`, modeling a 0/1 knapsack choice (either use `i` or not).
 * 4. The transition `if (flag[j - i] && dp[j] < dp[j - i] + 1)` checks if we can form a better partition for `j`. It asks: "If we can form a sum `j-i`, can we add `i` to it to get a longer partition for `j` than we've found so far?".
 * 5. If so, it updates `dp[j]`, sets `mp[j] = j - i` to remember the path, and marks `flag[j]` as true.
 * 6. After the DP table is populated, the program reads `n`, prints `dp[n]`, and then backtracks using the `mp` array to print the integers in the partition.
 *
 * Time Complexity: O(N^2) for the DP pre-computation.
 * Space Complexity: O(N) for the DP arrays.
 * Note: A simpler greedy approach with O(sqrt(n)) time complexity also solves this problem.
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <vector>

// Competitive programming template macros
#define endl "\n"
#define si(a) scanf("%d", &a)
#define sii(a, b) scanf("%d %d", &a, &b)
#define siii(a, b, c) scanf("%d %d %d", &a, &b, &c)
#define sl(a) scanf("%lld", &a)
#define sll(a, b) scanf("%lld %lld", &a, &b)
#define pf printf
#define vi vector<int>
#define vii vector<pair<int, int>>
#define pii pair<int, int>
#define all(c) c.begin(), c.end()
#define ms(a, c) memset(a, c, sizeof(a))
#define tr(c, i) for (decltype((c).begin()) i = (c).begin(); i != (c).end(); i++)
#define trr(c, i) for (decltype((c).rbegin()) i = (c).rbegin(); i != (c).rend(); i++)

using namespace std;

#define N 1001
#define mod 1000000007
#define inf 1e9

typedef long long ll;
typedef unsigned long long ull;

// Unused template function
int power(int a, int b) {
  int res = 1;
  while (b) {
    if (b & 1)
      res *= a;
    a *= a;
    b >>= 1;
  }
  return res;
}

int dp[N], mp[N];
bool flag[N];

int main() {
  ios_base::sync_with_stdio(0);
  cin.tie(0);

  // Pre-computation phase using Dynamic Programming.
  flag[0] = true; // Base case: a sum of 0 is possible.
  
  /**
   * Block Logic: Populate the DP table to find the optimal partitions.
   * This pre-calculates the maximum number of distinct integers that can sum up to any number `j` up to `N`.
   */
  for (int i = 1; i < N; i++) {       // Iterate through each possible distinct integer `i` to use.
    for (int j = N - 1; j - i >= 0; j--) { // Iterate downwards to ensure `i` is used at most once per sum.
      // If we can make sum `j-i`, and adding `i` gives a longer partition for `j`.
      if (flag[j - i] && dp[j] < dp[j - i] + 1)
        dp[j] = dp[j - i] + 1, mp[j] = j - i, flag[j] = true;
    }
  }
  
  int n;
  cin >> n;
  
  // Output the maximum number of children.
  cout << dp[n] << endl;
  
  /**
   * Block Logic: Backtrack using the `mp` array to reconstruct and print the partition.
   * Pre-condition: The DP table is filled, `n` holds the target sum.
   * Invariant: In each step, `n - mp[n]` reveals the last number added to the sum.
   */
  while (n)
    cout << n - mp[n] << ' ', n = mp[n];
  cout << endl;
  
  return 0;
}
