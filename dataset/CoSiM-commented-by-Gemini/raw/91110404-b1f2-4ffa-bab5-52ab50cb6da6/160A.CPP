/**
 * @file 160A.CPP
 * @brief Solution for the "Twins" problem (e.g., Codeforces 160A).
 * @details This program uses a greedy algorithm to find the minimum number of coins
 * a person needs to take from a set so that their sum is strictly greater than the
 * sum of the remaining coins. The strategy is to always pick the highest-value coins first.
 *
 * @algorithm Greedy Algorithm
 * @complexity
 * - Time: O(N log N), dominated by the sorting operation on the N coins.
 * - Space: O(N) to store the coin values in a vector.
 */

#include <bits/stdc++.h>

using namespace std;

// --- Common Competitive Programming Macros & Typedefs ---
typedef long long LL;
typedef vector<int> vi;
typedef vector<long long> vl;

#define MOD 1000000007
#define _for(i, n) for (int i = 0; i < n; i++)
#define __for(i, n) for (int i = 1; i <= n; i++)
#define REP(i, s, n) for (int i = (s), _n = (n); i <= _n; i++)
#define FOR(i, n) for (int i = 0, _n = (n); i < _n; i++)
#define REPD(i, e, s) for (int i = (e), _s = (s); i >= _s; i--)
#define tr(container, it)                                                      
  for (typeof(container.begin()) it = container.begin();                       
       it != container.end(); it++)
#define pb push_back
#define mp make_pair
#define fi first
#define se second
#define ALL(x) x.begin(), x.end()

int main() {
  int n; // Number of coins.
  int x; // Temporary variable for reading a coin's value.
  int sum1 = 0; // The total sum of all coins, later represents the remaining sum.
  int sum2 = 0; // The sum of coins taken.
  vector<int> v; // Vector to store coin values.

  scanf("%d", &n);

  // Read all coin values and calculate the total sum.
  for (int i = 0; i < n; i++) {
    scanf("%d", &x);
    v.pb(x);
    sum1 += x;
  }
  
  // Sort the coins in ascending order.
  sort(v.begin(), v.end());
  
  int i;
  /**
   * @block Greedy Selection
   * @logic Iterates backwards from the largest coin. In each step, it moves a coin
   * from the "remaining" pile (sum1) to the "taken" pile (sum2) until the
   * "taken" sum is strictly greater than the "remaining" sum.
   */
  for (i = v.size() - 1; i >= 0; i--) {
    sum1 -= v[i];
    sum2 += v[i];
    if (sum2 > sum1) {
      break;
    }
  }

  // The number of coins taken is the number of iterations performed.
  // This can be calculated from the final value of the loop counter `i`.
  printf("%d
", (int)v.size() - i);
  
  return 0;
}
