/**
 * @file 160A_2.CPP
 * @brief Solves the "Twins" problem by finding the minimum number of coins for a majority sum.
 * @details The goal is to determine the minimum number of coins one needs to take from a set
 * so that the sum of their values is strictly greater than the sum of the values of the
 * remaining coins. The strategy is to greedily take the coins with the highest values first.
 *
 * @section algorithm Algorithm
 * 1. Read the number of coins `n` and their respective values into an array.
 * 2. Sort the array of coins in non-decreasing order.
 * 3. The code then enters a loop that iteratively checks if taking a certain number of the
 *    largest coins satisfies the condition.
 * 4. It starts by checking the single largest coin, then the two largest, and so on.
 *    - A variable `k` tracks the partition between the smaller coins and the larger (taken) coins.
 *    - In each iteration, it calculates `sum1` (the sum of coins from index `k` to `n-1`) and
 *      `sum2` (the sum of coins from index `0` to `k-1`).
 *    - If `sum1` is strictly greater than `sum2`, the condition is met, and the loop terminates.
 *    - If not, `k` is decremented to include one more coin in the "taken" set, and the sums are
 *      recalculated in the next iteration.
 * 5. The final `count` of coins in the "taken" set is the answer.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(N log N + N^2) = O(N^2). The initial sort is O(N log N). The `while` loop can run up to N times, and inside it, the sums are recalculated in O(N) time, leading to an O(N^2) complexity for the search part.
 * - Space Complexity: O(N) to store the coin values in an array.
 */
#include <algorithm>
#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, ara[110], count = 0, sum1 = 0, sum2 = 0, j, i;
  cin >> n;
  for (i = 0; i < n; i++) {
    cin >> ara[i];
  }
  // Sort the coins in ascending order to make it easy to pick the largest ones.
  sort(ara, ara + n);
  int k = n - 1;

  /**
   * @brief This loop repeatedly partitions the coins and checks their sums.
   * It starts by considering taking only the largest coin and iteratively
   * adds more coins from the larger end until the sum of the taken coins
   * is greater than the sum of the remaining ones.
   */
  while (1) {
    // Calculate sum1 (our coins) and count. These are the coins from k to the end.
    for (i = n - 1; i >= k; i--) {
      sum1 += ara[i];
      count++;
    }
    // Calculate sum2 (the other twin's coins). These are the coins from the start up to k.
    for (j = 0; j < k; j++) {
      sum2 += ara[j];
    }
    // Check if we have taken enough coins to have a strictly larger sum.
    if (sum1 > sum2) {
      break;
    } else {
      // If not, we need to take at least one more coin.
      k--; // Move the partition to include another large coin.
      // Reset variables for the next iteration.
      count = 0;
      sum1 = 0;
      sum2 = 0;
    }
  }
  cout << count << endl;
}
