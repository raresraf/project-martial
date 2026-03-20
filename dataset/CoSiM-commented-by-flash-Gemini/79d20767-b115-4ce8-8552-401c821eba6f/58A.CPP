/**
 * @file 58A.CPP
 * @brief Solution to a problem involving distributing two types of items (red and blue).
 *        The problem likely aims to maximize the number of pairs formed from the two types,
 *        and then determine the remaining items. The output format suggests
 *        (number of pairs, remaining unpaired items of the dominant type).
 * Algorithm:
 *   1. Reads two integers, 'n' (red items) and 'm' (blue items).
 *   2. Employs a series of `if-else if` conditions to handle different scenarios based on:
 *      - Whether 'n' and 'm' are equal.
 *      - Whether one count is zero.
 *      - The relative sizes of 'n' and 'm'.
 *      - The parity (odd/even) of 'n' and 'm'.
 *   3. For each scenario, it calculates the number of pairs (which is `min(n, m)`)
 *      and the remaining difference (`|n - m| / 2`).
 *      The output is generally `min(n, m)` for the first value and `|n-m|/2` for the second value.
 *      The code explicitly handles various parities and relative sizes, which can be simplified
 *      to `min(n, m)` pairs and `(max(n, m) - min(n, m)) / 2` remaining.
 * Time Complexity: O(1) as it involves a fixed number of comparisons and arithmetic operations.
 * Space Complexity: O(1) for storing a few variables.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  /**
   * @brief Main function solving a distribution problem involving two types of items.
   *
   * Reads counts of two item types (n and m) and calculates the maximum number
   * of pairs that can be formed and the count of remaining unpaired items.
   * This is achieved through a detailed conditional logic that accounts for
   * various relationships between n and m, including their parity.
   * @return int Returns 0 upon successful execution.
   */
  int n, m;
  // n red m blue
  cin >> n >> m; // Reads the counts of red (n) and blue (m) items.

  // Block Logic: Handles various scenarios for item distribution based on counts and parity.
  // Each 'if-else if' block checks a specific condition and prints the number of pairs
  // and remaining unpaired items.
  // Pre-condition: 'n' and 'm' are non-negative integers.
  if (n == m) {
    // Case 1: Equal number of red and blue items.
    // All items can form pairs, no items left over.
    cout << n << " " << 0 << endl;
  } else if (n > m && m == 0) {
    // Case 2: Only red items, no blue items.
    // No pairs can be formed. All red items remain.
    // The problem statement likely implies something about groups of 2.
    cout << 0 << " " << n / 2 << endl;
  } else if (m > n && n == 0) {
    // Case 3: Only blue items, no red items.
    // No pairs can be formed. All blue items remain.
    cout << 0 << " " << m / 2 << endl;
  } else if (n > m && (n % 2 == 0) && (m % 2 == 0)) {
    // Case 4: More red items than blue, both counts are even.
    // All blue items pair with red items. The difference (n-m) is even.
    cout << m << " " << (n - m) / 2 << endl;
  } else if (m > n && (n % 2 == 0) && (m % 2 == 0)) {
    // Case 5: More blue items than red, both counts are even.
    // All red items pair with blue items. The difference (m-n) is even.
    cout << n << " " << (m - n) / 2 << endl;
  } else if (n > m && (n % 2 == 1) && (m % 2 == 1)) {
    // Case 6: More red items than blue, both counts are odd.
    // All blue items pair with red. One red item is left over. The difference (n-m) is even.
    cout << m << " " << (n - m) / 2 << endl;
  } else if (m > n && (n % 2 == 1) && (m % 2 == 1)) {
    // Case 7: More blue items than red, both counts are odd.
    // All red items pair with blue. One blue item is left over. The difference (m-n) is even.
    cout << n << " " << (m - n) / 2 << endl;
  } else if (n > m && (n % 2 == 1) && (m % 2 == 0)) {
    // Case 8: More red items than blue, n is odd, m is even.
    // All blue items pair with red. One red item is left over.
    int in = (n - m) / 2; // (n-m) is odd. Integer division floors the result.
    cout << m << " " << in << endl;
  } else if (n > m && (n % 2 == 0) && (m % 2 == 1)) {
    // Case 9: More red items than blue, n is even, m is odd.
    // All blue items pair with red. One red item is left over.
    int in = (n - m) / 2; // (n-m) is odd. Integer division floors the result.
    cout << m << " " << in << endl;
  } else if (m > n && (n % 2 == 0) && (m % 2 == 1)) {
    // Case 10: More blue items than red, n is even, m is odd.
    // All red items pair with blue. One blue item is left over.
    int in = (m - n) / 2; // (m-n) is odd. Integer division floors the result.
    cout << n << " " << in << endl;
  } else if (m > n && (n % 2 == 1) && (m % 2 == 0)) {
    // Case 11: More blue items than red, n is odd, m is even.
    // All red items pair with blue. One blue item is left over.
    int in = (m - n) / 2; // (m-n) is odd. Integer division floors the result.
    cout << n << " " << in << endl;
  }
}
