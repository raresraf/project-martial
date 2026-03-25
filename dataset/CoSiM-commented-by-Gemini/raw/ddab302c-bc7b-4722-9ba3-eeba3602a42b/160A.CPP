#include <iostream>
#include <stdlib.h>

/**
 * @file 160A.CPP
 * @brief Solution to a competitive programming problem, "Twins", using a
 * C-style approach within C++.
 *
 * Problem Statement:
 * Given a set of coins of varying values, find the minimum number of coins
 * you need to take such that the sum of your coins is strictly greater than
 * the sum of the remaining coins.
 *
 * Algorithm:
 * The approach is a variant of the standard greedy strategy. Instead of
 * building up a sum, it calculates a target value and then determines how
 * many of the largest coins are needed to meet or exceed that target.
 *
 * 1. Calculate the total sum of all coin values.
 * 2. Determine the target sum. To have a sum strictly greater than the rest,
 *    one needs a sum of at least `floor(total_sum / 2) + 1`. The code
 *    calculates this target, using a flag to handle the edge case where the
 *    total sum is even.
 * 3. Sort the coins in ascending order using C's `qsort`.
 * 4. Iterate from the largest coin downwards, subtracting its value from the
 *    target sum and counting how many coins are "used" until the target is
 *    met (i.e., the remaining target sum is <= 0).
 * 5. A final adjustment is made for the case where the total sum was even and
 *    the coins taken sum up to exactly half the total, requiring one more coin.
 *
 * Time Complexity: O(N log N) due to the `qsort` operation.
 * Space Complexity: O(N) to store the coin values.
 */

// Comparison function for qsort, sorts in ascending order.
int compare(const void *a, const void *b) { return (*(int *)a - *(int *)b); }

int main() {
  int flag = 0; // Flag to indicate if the total sum is even.
  int i, n;
  int value[105];   // Array to hold coin values.
  int sum = 0;      // Initially holds total sum, then the target sum.
  int num = 0;      // Counter for the number of coins taken.

  std::cin >> n;

  // Read coin values and calculate the total sum.
  for (i = 0; i < n; i++) {
    std::cin >> value[i];
    sum = sum + value[i];
  }

  // Block Logic: Calculate the minimum sum required to win.
  // This is floor(total/2) + 1. The code calculates this in a complex way.
  if (sum % 2 == 0) {
    // If total is even, target is total/2. We need to strictly exceed this.
    sum = sum / 2;
    flag = 1;
  } else {
    // If total is odd, target is (total/2)+1, which is what we need to reach.
    sum = (sum + 1) / 2;
  }

  // Sort the coins to easily access the largest ones first.
  qsort(value, n, sizeof(int), compare);

  // Block Logic: Greedily take the largest coins.
  // 'sum' is now the target value to "pay off".
  // Invariant: 'num' is the count of the largest coins taken so far.
  for (i = n - 1; i >= 0; i--) {
    // Pre-condition: Check if there's still a target sum to meet.
    if (sum >= value[i]) {
      // If the remaining target is greater than the current coin,
      // "use" the coin and decrement the target.
      num++;
      sum = sum - value[i];
    } else if (sum < value[i]) {
      // If the current coin alone exceeds the remaining target.
      if (sum != 0) {
        num++; // Take this one last coin.
      }
      break; // We have taken enough coins.
    }
  }

  // Post-processing: Handle the edge case where the sum of coins taken is
  // exactly half of an even total sum. We need one more coin to be strictly
  // greater.
  if (flag == 1 && sum == 0) {
    num++;
  }

  std::cout << num << '\n';
  return 0;
}