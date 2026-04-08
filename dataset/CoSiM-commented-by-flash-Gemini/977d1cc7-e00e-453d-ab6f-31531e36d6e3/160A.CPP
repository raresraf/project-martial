/**
 * @file 160A.CPP
 * @brief Solves the "Twins" problem from Codeforces, which is a classic greedy algorithm problem.
 * The goal is to find the minimum number of coins to take so that their sum is strictly greater 
 * than the sum of the remaining coins.
 *
 * @algorithm
 * 1. Read the number of coins and their values.
 * 2. Calculate the total sum of all coin values.
 * 3. Sort the coins in non-decreasing order.
 * 4. Iterate from the largest coin downwards, accumulating the sum of coins taken (`now`).
 * 5. In each iteration, increment the count of coins taken (`ans`).
 * 6. Stop when the sum of coins taken (`now`) is strictly greater than the sum of the remaining coins (`tot - now`).
 * 7. The final count `ans` is the minimum number of coins required.
 *
 * @time_complexity O(N log N) - dominated by the sorting of the coins.
 * @space_complexity O(N) - to store the coin values.
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <queue>
#include <stack>
#include <string>
#include <utility>
#include <vector>

#define fi first
#define se second
#define pb push_back
#define mp make_pair
#define pi 2 * acos(0.0)
#define eps 1e-9
#define PII pair<int, int>
#define PDD pair<double, double>
#define LL long long
#define INF 1000000000

using namespace std;

int N; // Number of coins
int x, y, z; // Loop counters and general purpose variables
int tot; // Total sum of all coin values
int ans; // The result: minimum number of coins to take
int now; // The current sum of coins taken
int coin[111]; // Array to store the value of each coin

int main() {
  // Read the number of coins.
  scanf("%d", &N);

  /**
   * @block_logic Read the coin values and calculate the total sum.
   * @pre N is the number of coins to read.
   * @post The `coin` array is populated, and `tot` holds the sum of all coin values.
   */
  tot = 0;
  for (x = 1; x <= N; x++) {
    scanf("%d", &coin[x]);
    tot += coin[x];
  }

  // Sort the coins in ascending order to apply a greedy strategy.
  sort(coin + 1, coin + 1 + N);

  /**
   * @block_logic Greedily take the largest coins until their sum exceeds the rest.
   * @pre `coin` array is sorted, `tot` is the total sum.
   * @invariant `now` is the sum of the `ans` largest coins.
   * @post The loop breaks when `now > tot - now`, meaning the sum of taken coins
   *       is greater than the sum of the remaining ones. `ans` holds the count.
   */
  ans = now = 0;
  for (x = N; x > 0; x--) {
    now += coin[x]; // Add the largest available coin to our set.
    ans++; // Increment the count of coins taken.

    // Check if the sum of our coins is now greater than the sum of the remaining coins.
    if (now > tot - now)
      break;
  }

  // Print the minimum number of coins required.
  printf("%d\n", ans);
  return 0;
}
