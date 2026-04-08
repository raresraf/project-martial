/**
 * @file 160A.CPP
 * @brief An attempted solution for the "Twins" problem from Codeforces (Problem 160A).
 * @details The program tries to find the minimum number of coins to take so their
 * sum is strictly greater than the sum of the remaining coins.
 *
 * @algorithm
 * This solution uses a greedy but convoluted approach.
 * 1. Read all coin values and calculate the total sum.
 * 2. Calculate a `target_sum`. The logic aims for the smallest integer sum that is
 *    strictly greater than `total_sum / 2`.
 * 3. Sort the coins in non-decreasing order using `qsort`.
 * 4. Greedily iterate from the largest coins downwards, subtracting their value
 *    from the `target_sum` and counting the number of coins taken. The goal is
 *    to reduce the `target_sum` to zero or less.
 * 5. A special check is performed after the loop to handle the case where the
 *    initial total sum was even and the taken coins sum to exactly half of it.
 *    In this case, one more coin is needed to make the sum strictly greater,
 *    so the count is incremented.
 *
 * @note This implementation is complex and its logic for meeting the target sum
 * is less direct than simply accumulating a sum and comparing. The final flag-based
 * check is a patch for this less robust approach.
 *
 * @time_complexity O(N log N) due to the sorting step.
 * @space_complexity O(N) to store the coin values.
 */
#include <iostream>
#include <stdlib.h>

// C-style comparison function for qsort.
int compare(const void *a, const void *b) { return (*(int *)a - *(int *)b); }

int main() {
  int flag = 0; // Flag to check if the initial sum is even.
  int i, n, value[105], sum = 0, num = 0;
  std::cin >> n;
  for (i = 0; i < n; i++) {
    std::cin >> value[i];
    sum = sum + value[i];
  }

  /**
   * @block_logic Calculate the minimum sum required to be strictly greater
   * than the other half. This is equivalent to floor(total_sum / 2) + 1.
   */
  if (sum % 2 == 0) {
    sum = sum / 2;
    flag = 1;
  } else
    sum = (sum + 1) / 2;

  // Sort the coins to enable a greedy approach.
  qsort(value, n, sizeof(int), compare);

  /**
   * @block_logic Greedily take the largest coins. Instead of building a sum up,
   * this loop subtracts from the target sum until it reaches zero or less.
   */
  for (i = n - 1; i >= 0; i--) {
    if (sum >= value[i]) {
      num++;
      sum = sum - value[i];
    } else if (sum < value[i]) {
      // If a single coin is greater than the remaining sum needed,
      // taking it is enough.
      if (sum != 0)
        num++;
      break;
    }
  }

  // Edge case: if the total sum was even and we took exactly half,
  // we need one more coin to be *strictly* greater.
  if (flag == 1 && sum == 0)
    num++;
  
  std::cout << num << '\n';
  return 0;
}
