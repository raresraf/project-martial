/**
 * @file 160A.CPP
 * @brief Solution for the "Twins" problem (Codeforces 160A).
 * @author competitive-programmer
 *
 * @details The problem requires finding the minimum number of coins to take from a
 * set so that their sum is strictly greater than the sum of the remaining coins.
 *
 * Algorithm: Greedy Approach
 * 1. Read all coin values and calculate their total sum (`jami`).
 * 2. The goal is to collect a subset of coins whose sum is strictly greater than
 *    `jami / 2`.
 * 3. Sort the coins in ascending order.
 * 4. Iterate backward from the largest coin.
 * 5. Greedily take the largest coins one by one, adding their value to a running
 *    `sum` and incrementing a counter `num`.
 * 6. Continue this process as long as the `sum` of coins taken is less than or
 *    equal to half of the total sum.
 * 7. The final value of `num` is the minimum number of coins required.
 *
 * Time Complexity: O(N log N), dominated by the sorting operation, where N is the
 * number of coins.
 * Space Complexity: O(N) to store the coin values.
 */
#include <algorithm>
#include <iostream>
using namespace std;

// Global variables for number of coins, coin values, sums, and counts.
int a, mas[105], sum, num, jami;

int main() {
  // `a` is the number of coins.
  cin >> a;
  /**
   * Block Logic: Read all coin values and calculate the total sum (`jami`).
   */
  for (int i = 0; i < a; i++) {
    cin >> mas[i];
    jami += mas[i];
  }
  
  // Sort the coins in non-decreasing (ascending) order.
  sort(mas, mas + a);
  
  /**
   * Block Logic: Greedily take the largest coins until their sum is greater
   * than the sum of the remaining coins.
   * The loop iterates backward, from the largest coin to the smallest.
   */
  for (int i = (a - 1); i >= 0; i--) {
    // Condition: Continue taking coins as long as our share (`sum`) is not
    // strictly greater than the other twin's share (`jami / 2`).
    if (sum <= jami / 2) {
      sum += mas[i]; // Take the coin.
      num++;         // Increment the count of coins taken.
    }
    // Note: A `break` here would make the loop more efficient once the condition is met.
  }
  
  // Print the minimum number of coins required.
  cout << num;
}
