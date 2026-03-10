/**
 * @file 160A.CPP
 * @brief Solution for a competitive programming problem (Codeforces "Twins").
 * The goal is to find the minimum number of coins to take from a set such that
 * the sum of the taken coins is strictly greater than the sum of the remaining coins.
 *
 * @algorithm Greedy Approach. To minimize the number of coins while maximizing their
 * sum, it's always optimal to take the coins with the largest values first.
 *
 * @time_complexity O(N log N), dominated by the sorting of the coins.
 * @space_complexity O(N) to store the values of the N coins.
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

// Macros for common data structures and constants
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

int N, x, y, z, tot, ans, now;
int coin[111]; // Array to store coin values

int main() {
  scanf("%d", &N);

  /**
   * @block
   * @brief Reads coin values and calculates the total sum.
   */
  tot = 0;
  for (x = 1; x <= N; x++) {
    scanf("%d", &coin[x]);
    tot += coin[x];
  }

  // Sorts the coins in non-decreasing order to facilitate the greedy approach.
  sort(coin + 1, coin + 1 + N);

  ans = now = 0;
  /**
   * @block
   * @brief Greedily takes the largest coins until the sum is sufficient.
   *
   * Invariant: At each step, `now` holds the sum of the `ans` largest coins.
   * The loop iterates from the largest coin downwards, adding its value to `now`
   * and incrementing the coin count `ans`. It stops as soon as the sum of
   * taken coins (`now`) exceeds the sum of the remaining coins (`tot - now`).
   */
  for (x = N; x > 0; x--) {
    now += coin[x];
    ans++;

    if (now > tot - now)
      break;
  }

  printf("%d\n", ans);
  return 0;
}
