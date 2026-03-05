/**
 * @file 160A.CPP
 * @brief Solution to the "Twins" problem (Codeforces 160A).
 *
 * This program solves a problem where two twins want to divide a set of coins
 * such that one twin's sum of coins is strictly greater than the other's.
 * The goal is to find the minimum number of coins a twin needs to take
 * to achieve this condition.
 *
 * Algorithm: Greedy approach. Sort the coins in descending order. Then,
 * iteratively pick the largest coins until their accumulated sum strictly
 * exceeds half of the total sum of all coins.
 * Time Complexity: O(N log N) due to sorting, where N is the number of coins.
 * Space Complexity: O(N) to store the coins.
 */
#include <algorithm>
#include <iostream>
using namespace std;

/**
 * @brief Main function to find the minimum number of coins for one twin
 *        to have a sum strictly greater than half the total.
 *
 * Reads the number of coins (n) and their values from standard input.
 * It then applies a greedy algorithm by sorting the coins in descending order
 * and iteratively picking the largest ones until the sum exceeds half of the total.
 *
 * @return 0 on successful execution.
 */
int main() {
  // Declare integer 'n' for the number of coins, and 'sum' for the total sum of all coins.
  int n, sum = 0;
  // Read the number of coins 'n' from standard input.
  cin >> n;
  // Declare loop counter 'i' and an array 'a' of size 'n' to store coin values.
  int i, a[n];
  /**
   * @brief Block Logic: Reads coin values and calculates their total sum.
   *
   * Pre-condition: 'n' is the number of coins to be read.
   * Invariant: 'sum' accumulates the value of each coin read.
   */
  for (i = 0; i < n; i++) {
    cin >> a[i];
    sum += a[i];
  }
  // Sort the 'a' array in descending order to apply the greedy strategy.
  sort(a, a + n, greater<int>());
  // 'coins' counts the number of coins taken, 'taken' stores their sum.
  int coins = 0, taken = 0;
  // Take the first (largest) coin initially.
  taken = a[0];
  coins = 1;
  i = 1; // Start checking from the second largest coin.
  int j = n - 1; // Represents the last index of the array (smallest coin).
  // Reduce the total sum by the first coin taken to update the remaining sum.
  sum -= a[0];
  /**
   * @brief Block Logic: Greedily selects coins starting from the largest until
   *        their sum is strictly greater than half of the total sum.
   *
   * Iterates through the sorted array, adding coins to 'taken' and updating 'sum'.
   * Pre-condition: 'a' is sorted in descending order, 'taken' has the largest coin,
   * 'sum' is the remaining total sum, and 'coins' is 1.
   * Invariant: 'taken' increases, 'sum' decreases, and 'coins' counts selected coins.
   */
  while (i <= j) {
    // If the sum of 'taken' coins already exceeds the remaining 'sum', break.
    if (taken > sum)
      break;
    // Add the next largest coin to 'taken'.
    taken = taken + a[i];
    // Subtract the added coin from the 'sum' of remaining coins.
    sum = sum - a[i];
    // Increment the count of selected coins.
    coins++;
    i++; // Move to the next largest coin.
  }
  // Print the minimum number of coins required.
  cout << coins;
}