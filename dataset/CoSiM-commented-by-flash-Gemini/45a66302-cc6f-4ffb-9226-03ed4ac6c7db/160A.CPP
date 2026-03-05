/**
 * @file 160A.CPP
 * @brief Solution to the "Twins" problem (Codeforces 160A).
 *
 * This program solves a problem where two twins want to divide a set of coins
 * such that one twin's sum of coins is strictly greater than the other's.
 * The goal is to find the minimum number of coins a twin needs to take
 * to achieve this condition.
 *
 * Algorithm: Greedy approach. Sort the coins in descending order and iteratively
 * pick the largest coins until their sum exceeds half of the total sum of all coins.
 * Time Complexity: O(N log N) due to sorting, where N is the number of coins.
 * Space Complexity: O(N) to store the coins.
 */
#include <bits/stdc++.h>

using namespace std;

/**
 * @brief Main function to find the minimum number of coins whose sum
 *        is strictly greater than half of the total sum.
 *
 * Reads the number of coins (n) and their values from standard input.
 * It then applies a greedy algorithm by sorting the coins and picking
 * the largest ones until the sum exceeds half of the total.
 *
 * @return 0 on successful execution.
 */
int main() {
  // Declare integers for the number of coins 'n', loop counter 'i', and total sum 'sum'.
  int n, i, sum = 0;
  // Declare an array 'a' of size 100, initialized to 0. (Note: This array is unused in the current logic.)
  int a[100];
  // Initialize 'a' to all zeros. (This loop is unnecessary as 'a' is not used later.)
  for (i = 1; i <= 100; i++)
    a[i] = 0;
  // Read the number of coins 'n' from standard input.
  cin >> n;
  // Declare an array 'arr' of size 'n' to store the coin values.
  int arr[n];
  /**
   * @brief Block Logic: Reads coin values and calculates their total sum.
   *
   * Pre-condition: 'n' is the number of coins to be read.
   * Invariant: 'sum' accumulates the value of each coin read.
   */
  for (i = 0; i < n; i++) {
    cin >> arr[i];
    sum += arr[i];
  }
  // Calculate half of the total sum. The goal is to exceed this value.
  int div = sum / 2;
  // Sort the 'arr' array in ascending order.
  sort(arr, arr + n);
  // Initialize 's' to store the sum of selected coins and 'coins' to count them.
  int s = 0;
  int coins = 0;
  /**
   * @brief Block Logic: Greedily selects coins starting from the largest until
   *        their sum is strictly greater than half of the total sum.
   *
   * Iterates through the sorted array from the largest coin to the smallest.
   * Pre-condition: 'arr' is sorted in ascending order.
   * Invariant: 's' increases with each selected coin, and 'coins' counts them.
   */
  for (i = n - 1; i >= 0; i--) {
    // If the current sum 's' already exceeds 'div', then no more coins are needed.
    if (s > div)
      break;
    else {
      // Add the current largest coin to 's'.
      s += arr[i];
      // Increment the count of selected coins.
      coins++;
    }
  }
  // Print the minimum number of coins required.
  cout << coins;
  return 0;
}
