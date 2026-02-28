/**
 * @file 160A.CPP
 * @brief A flawed C implementation for the "Twins" problem from Codeforces.
 * @details The goal is to find the minimum number of coins to take from a set so their
 * sum is strictly greater than the sum of the remaining coins. This code attempts
 * to solve this with a greedy approach but contains critical bugs.
 *
 * @algorithm Intended Greedy Algorithm
 * 1. Read all coin values and calculate their total sum `S`.
 * 2. Sort the coins in ascending order using bubble sort.
 * 3. Greedily take the largest coins first by iterating backwards from the end of
 *    the sorted array.
 * 4. Keep a running sum `k` of the taken coins and a count of how many coins were taken.
 * 5. Stop when the sum of taken coins `k` is strictly greater than the sum of the
 *    remaining coins (`S - k`).
 *
 * @bugs
 * 1. **Uninitialized Variable:** The variable `k` (sum of taken coins) is used in
 *    the expression `k += ...` without being initialized, leading to undefined behavior.
 *    It should be initialized to 0.
 * 2. **Off-by-One Error:** The `count` of coins is initialized to 1 instead of 0.
 *    This means the final count will always be at least one greater than the correct answer.
 *
 * @complexity
 * - Time: O(N^2), dominated by the bubble sort implementation.
 * - Space: O(N), to store the coin values in an array.
 */
#include <stdio.h>
#include <string.h>

int main() {
  // n: number of coins. a: array to store coin values. i, j: loop counters.
  // S: total sum of all coins. k: sum of coins taken. count: number of coins taken.
  int n, a[100], i, j, S = 0, k, count = 1;
  
  // BUG: `k` is not initialized. It should be `int k = 0;`.
  // BUG: `count` is initialized to 1. It should be 0.
  
  scanf("%d", &n);
  // Read all coin values and calculate the total sum `S`.
  for (i = 0; i < n; i++) {
    scanf("%d", &a[i]);
    S += a[i];
  }
  
  /**
   * @brief Sort the array in ascending order using bubble sort.
   * The swap logic uses arithmetic operations instead of a temp variable.
   */
  for (i = 0; i < n; i++) {
    for (j = 0; j < n - 1; j++)
      if (a[j] > a[j + 1]) {
        a[j] = a[j] + a[j + 1];
        a[j + 1] = a[j] - a[j + 1];
        a[j] = a[j] - a[j + 1];
      }
  }

  /**
   * @brief Greedily take the largest coins until the sum is sufficient.
   * This loop iterates backwards through the sorted (ascending) array to get
   * the largest elements first.
   */
  for (i = 0; i < n; i++) {
    // Add the next largest coin to the sum of taken coins `k`.
    k += a[n - i - 1];
    
    // Check if the sum of taken coins is still less than or equal to the remaining sum.
    if (k <= S - k)
      count++; // If so, we need to take another coin.
    else
      break; // Otherwise, we have taken enough coins, so we stop.
  }
  
  // Print the final count, which will be incorrect due to the bugs.
  printf("%d", count);
  
  return 0;
}
