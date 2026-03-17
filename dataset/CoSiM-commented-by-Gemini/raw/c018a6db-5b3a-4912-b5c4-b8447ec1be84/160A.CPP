/**
 * @file 160A.CPP
 * @brief Solution to a competitive programming problem, likely Codeforces 160A "Twins".
 *
 * @details
 * This program solves a classic greedy problem. Given a set of coins of varying
 * values, the objective is to determine the minimum number of coins to take such
 * that their sum is strictly greater than the sum of the remaining coins.
 *
 * @algorithm
 * The optimal strategy is a greedy one:
 * 1. Read all coin values and simultaneously calculate their total sum.
 * 2. Sort the coins in descending order to easily access the largest values.
 * 3. Iterate through the sorted coins, taking the largest ones first and adding
 *    them to a personal sum.
 * 4. After each coin is taken, check if the personal sum has become strictly
 *    greater than the sum of the remaining coins.
 * 5. The process stops as soon as this condition is met, and the number of coins
 *    taken is the answer.
 *
 * @complexity
 * - Time Complexity: O(N log N), where N is the number of coins, dominated by the
 *   sorting operation.
 * - Space Complexity: O(N) to store the coin values in an array.
 */
#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function and entry point of the program.
 */
int main() {
  int n; // Number of coins.
  int m = 0; // Total sum of all coin values.
  int p = 0; // Sum of coins taken by you.
  int i = 0; // Counter for the number of coins taken.
  
  cin >> n;
  int a[n + 1];
  
  // Read coin values and calculate the total sum.
  for (int i = 0; i < n; i++) {
    cin >> a[i];
    m = m + a[i];
  }
  
  // Sort the coins in descending (greater to smaller) order.
  sort(a, a + n, greater<int>());

  /**
   * @brief Greedily take the largest coins until your sum is greater.
   * The loop iterates, taking one coin at a time, until the condition is met.
   * The loop counter `i` also serves as the count of coins taken.
   */
  for (i = 0; i < n; i++) {
    // The sum of remaining coins is `m - p`. Check if your sum `p` is greater.
    if (p > m - p)
      break;
    else
      // Take the next largest coin.
      p = p + a[i];
  }
  
  // `i` holds the number of coins taken when the loop broke.
  cout << i << endl;
}
