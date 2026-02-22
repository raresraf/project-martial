/**
 * @file 160A.CPP
 * @brief Solution for Codeforces Problem 160A: Twins.
 * @details This program solves a problem where it's required to take the minimum number of coins
 * (represented by integers) such that their sum is strictly greater than the sum of the remaining coins.
 * It reads 'n' coin values, calculates their total sum, sorts the coins in descending order,
 * and then iteratively sums the largest coins until their sum exceeds half of the total sum.
 * The count of these coins is then printed.
 * @algorithm Greedy approach with sorting.
 * Time Complexity: $O(N \log N)$ due to sorting, where N is the number of coins.
 * Space Complexity: $O(N)$ to store the coin values.
 */
#include <algorithm>
#include <cstdio>
using namespace std;
bool cmp(int a, int b) { return a > b; }
/**
 * @file 160A.CPP
 * @brief Solution for Codeforces Problem 160A: Twins.
 * @details This program solves a problem where it's required to take the minimum number of coins
 * (represented by integers) such that their sum is strictly greater than the sum of the remaining coins.
 * It reads 'n' coin values, calculates their total sum, sorts the coins in descending order,
 * and then iteratively sums the largest coins until their sum exceeds half of the total sum.
 * The count of these coins is then printed.
 * @algorithm Greedy approach with sorting.
 * Time Complexity: $O(N \log N)$ due to sorting, where N is the number of coins.
 * Space Complexity: $O(N)$ to store the coin values.
 */
#include <algorithm> // Required for std::sort.
#include <cstdio>    // Required for scanf and printf.
using namespace std; // Using the standard namespace to avoid prefixing std::.

/**
 * @brief Custom comparison function for sorting in descending order.
 * @param a First integer for comparison.
 * @param b Second integer for comparison.
 * @return True if `a` is greater than `b`, false otherwise.
 */
bool cmp(int a, int b) { return a > b; }

int main() {
  /**
   * @brief Main function: orchestrates the coin selection logic.
   */
  int n,         // @var n: The total number of coins.
      a[101],    // @var a: Array to store the coin values (up to 100 coins, 1-indexed usage).
      i,         // @var i: Loop counter and final count of selected coins.
      x = 0,     // @var x: Stores the total sum of all coin values.
      y = 0;     // @var y: Stores the sum of selected coin values.

  scanf("%d", &n); // Reads the total number of coins.

  /**
   * @brief Reads coin values and calculates their total sum.
   * @block_logic Iterates `n` times to read each coin value and add it to `x`.
   * @invariant `a[i]` stores the i-th coin value, and `x` accumulates the total sum.
   * @note The array `a` is used 1-indexed, meaning `a[0]` is unused.
   */
  for (i = 1; i <= n; i++) {
    scanf("%d", &a[i]);
    x += a[i]; // Accumulates the total sum of coins.
  }

  /**
   * @brief Sorts the coin values in descending order.
   * @details This is a crucial step for the greedy approach, ensuring that the largest coins are considered first.
   * @param a + 1 Pointer to the second element of the array (start of 1-indexed data).
   * @param a + n + 1 Pointer one past the last element of the array (end of 1-indexed data).
   * @param cmp Custom comparison function to sort in descending order.
   */
  sort(a + 1, a + n + 1, cmp);

  i = 0; // Reset 'i' to count the selected coins.

  /**
   * @brief Greedily selects coins starting from the largest until their sum exceeds half of the total sum.
   * @block_logic Iteratively adds the largest remaining coins to `y` and increments `i`.
   * @invariant `y` is the sum of `i` largest coins, and `y` is less than or equal to half of the total sum `x`.
   */
  while (y <= (x) / 2) { // Loop continues as long as the sum of selected coins is not strictly greater than half.
    i++;                 // Increments the count of selected coins.
    y += a[i];           // Adds the next largest coin to the sum of selected coins.
  }

  // Functional Utility: Prints the minimum number of coins required.
  printf("%d", i); // Output the count of coins.
  return 0;        // Indicates successful execution.
}

