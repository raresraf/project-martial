/**
 * @file 160A.CPP
 * @brief Solution to Codeforces problem 160A - Twins.
 *
 * @algorithm
 * The problem asks for the minimum number of coins to take so that the sum of
 * their values is strictly greater than the sum of the remaining coins.
 *
 * Let the total sum of all coins be `S`. We want to find the smallest set of
 * coins whose sum `s_my` is greater than the sum of the remaining coins,
 * `s_twin = S - s_my`.
 * The condition is `s_my > S - s_my`, which simplifies to `s_my > S / 2`.
 *
 * To achieve this sum with the minimum number of coins, a greedy approach is
 * optimal:
 * 1. Calculate the total sum `S` of all coins.
 * 2. Sort the coins in descending order.
 * 3. Iterate through the sorted coins (from largest to smallest), adding their
 *    value to a running sum `s_my` and counting the number of coins taken.
 * 4. Stop as soon as `s_my` becomes strictly greater than `S / 2`.
 * 5. The number of coins taken is the answer.
 *
 * This implementation sorts in ascending order and then iterates backwards.
 *
 * @complexity
 * - Time Complexity: O(N log N), dominated by the sorting of the coins.
 * - Space Complexity: O(N) to store the coin values.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  // n: the number of coins.
  // i: loop counter.
  // sum: the total value of all coins.
  int n, i, sum = 0;
  
  // a: an unused array of size 100.
  int a[100];
  for (i = 1; i <= 100; i++)
    a[i] = 0;
  
  cin >> n;
  // arr: stores the values of the `n` coins.
  int arr[n];
  for (i = 0; i < n; i++) {
    cin >> arr[i];
    sum += arr[i];
  }

  // div: the target sum to exceed. We need a sum strictly greater than half the total.
  int div = sum / 2;
  
  // Sort the coins in non-decreasing (ascending) order.
  sort(arr, arr + n);
  
  // s: the running sum of the coins we take.
  int s = 0;
  // coins: the number of coins we have taken.
  int coins = 0;

  /**
   * Block Logic: Greedily take the largest coins until our sum is greater than half the total.
   * We iterate from the end of the sorted array to get the largest coins first.
   * Pre-condition: `arr` is sorted, `div` is sum/2.
   * Invariant: `s` is the sum of the `coins` largest coins.
   */
  for (i = n - 1; i >= 0; i--) {
    // If our sum `s` is already greater than the twin's share, we can stop.
    if (s > div)
      break;
    else {
      // Otherwise, take the next largest coin.
      s += arr[i];
      coins++;
    }
  }
  
  // Output the minimum number of coins required.
  cout << coins;
  return 0;
}
