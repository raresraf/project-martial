/**
 * @file 160A.CPP
 * @brief Solution for a greedy algorithm problem, likely from a programming contest.
 * @details The problem is to find the minimum number of items (e.g., coins) to take from a set
 * such that the sum of the values of the taken items is strictly greater than the sum of the
 * values of the remaining items.
 *
 * Algorithm:
 * This problem can be solved using a greedy approach. To minimize the number of items taken while
 * maximizing their sum, one should always pick the items with the highest values first.
 * 1. Read all item values and calculate their total sum.
 * 2. The target sum to exceed is `total_sum / 2`.
 * 3. Sort the items in ascending order to easily access the highest-value items.
 * 4. Greedily take items with the highest value one by one, adding their value to a running sum.
 * 5. Keep track of the number of items taken.
 * 6. Stop when the running sum is strictly greater than `total_sum / 2`.
 * 7. The number of items taken at this point is the minimum required.
 *
 * Time Complexity: O(n log n) dominated by the sorting step.
 * Space Complexity: O(n) to store the values of the items.
 */
// Written By Deepesh Nimma
// Problem: 160A Codeforces
#include <bits/stdc++.h>
using namespace std;

#define ll long long
#define space " "
#define end1 "\n"
#define ar array

int n, i, a[100], sum = 0, ans = 0, cnt = 0;

int main() {
  cin >> n;

  for (i = 0; i < n; ++i) {
    cin >> a[i];
  }

  // Sort the items in non-decreasing order to facilitate the greedy approach.
  sort(a, a + n);

  /**
   * @block
   * @description Calculate the total sum of all item values.
   */
  for (i = 0; i < n; ++i) {
    sum += a[i];
  }

  // The target is to have a sum strictly greater than half of the total sum.
  sum = sum / 2;

  /**
   * @block
   * @description Greedily take the items with the highest value until the
   * collected sum (`ans`) exceeds the sum of the remaining items.
   * Invariant: `ans` is the sum of the `cnt` largest items.
   */
  while (ans <= sum) {
    ++cnt;
    // Add the next largest item to the collected sum.
    ans += a[n - cnt];
  }

  // Output the minimum number of items required.
  cout << cnt;
}