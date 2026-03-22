/**
 * @file 160A.CPP
 * @brief Solves the "Twins" problem using a greedy approach.
 *
 * The problem is to find the minimum number of coins one needs to take from a set
 * so that the sum of their values is strictly greater than the sum of the remaining coins.
 *
 * @b Algorithm: Greedy Strategy
 *   The optimal strategy is to always pick the coins with the largest values.
 *   The code implements this by:
 *   1. Sorting all coin values in ascending order.
 *   2. Calculating half of the total sum.
 *   3. Iteratively taking the largest remaining coins until their sum exceeds half of the total.
 * @b Time_Complexity: O(N log N), dominated by the sorting step.
 * @b Space_Complexity: O(N) to store the coin values.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  // n: number of coins.
  // a: array to store coin values.
  // sum: total value of all coins.
  // ans: sum of coins taken.
  // cnt: number of coins taken.
  int n, i, a[100], sum = 0, ans = 0, cnt = 0;
  scanf("%d", &n);
  for (i = 0; i < n; i++) {
    scanf("%d", &a[i]);
  }
  
  // Sort the coins in non-decreasing order to easily access the largest ones.
  sort(a, a + n);
  
  // Calculate the total sum of all coin values.
  for (i = 0; i < n; i++) {
    sum += a[i];
  }
  
  // The target is to have a sum strictly greater than the sum of the rest.
  // This is equivalent to having a sum greater than half of the total sum.
  sum = sum / 2;
  
  /**
   * @brief Greedily take the largest coins until their sum exceeds the target.
   * Invariant: `ans` holds the sum of the `cnt` largest coins.
   * The loop continues as long as our sum (`ans`) is not strictly greater than
   * the sum of the remaining coins (`sum`).
   */
  while (ans <= sum) {
    ++cnt;
    ans += a[n - cnt];
  }
  
  // Print the minimum number of coins required.
  printf("%d\n", cnt);
  return 0;
}
