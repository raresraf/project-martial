/**
 * @file 160A.CPP
 * @brief Solution for a competitive programming problem (Codeforces "Twins").
 * The goal is to find the minimum number of coins to take from a set such that
 * the sum of the taken coins is strictly greater than the sum of the remaining coins.
 *
 * @algorithm Greedy Approach. To minimize the number of coins while maximizing
 * their sum, it is always optimal to take the coins with the largest values first.
 * The program sorts the coins in descending order and picks them one by one
 * until their sum exceeds half of the total sum.
 *
 * @time_complexity O(N log N), dominated by the sorting of the coins.
 * @space_complexity O(N) to store the values of the N coins.
 */
#include <algorithm>
#include <cstdio>
using namespace std;

/**
 * @brief Custom comparison function for sorting in descending order.
 */
bool cmp(int a, int b) { return a > b; }

int main() {
  int n, a[101], i;
  int x = 0; // x will store the total sum of all coins.
  int y = 0; // y will store the sum of coins taken.

  scanf("%d", &n);
  /**
   * @block
   * @brief Reads coin values and calculates the total sum.
   */
  for (i = 1; i <= n; i++) {
    scanf("%d", &a[i]);
    x += a[i];
  }

  // Sorts the coins in descending order to facilitate the greedy approach.
  sort(a + 1, a + n + 1, cmp);

  i = 0; // 'i' will count the number of coins taken.
  /**
   * @block
   * @brief Greedily takes the largest coins until the sum is sufficient.
   *
   * Invariant: `y` holds the sum of the `i` largest coins.
   * The loop continues as long as the sum of coins taken (`y`) is not strictly
   * greater than the sum of the remaining coins. The condition `y <= x / 2` is
   * equivalent to `y <= x - y`.
   */
  while (y <= (x) / 2) {
    i++;
    y += a[i];
  }
  printf("%d", i);
}
