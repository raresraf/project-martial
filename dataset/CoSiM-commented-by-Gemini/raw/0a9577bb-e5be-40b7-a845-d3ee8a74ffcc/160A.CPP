/**
 * @file 160A.CPP
 * @brief Solves the "Twins" problem from competitive programming.
 * @details The problem is to find the minimum number of coins to take such that their total value is strictly greater than the total value of the remaining coins.
 * This implementation iteratively selects the largest available coins until their sum exceeds half of the total sum of all coins.
 *
 * @algorithm Greedy approach by iteratively picking the largest coins.
 * @complexity
 * Time complexity: O(N * M) in the worst case, where N is the number of coins and M is the maximum coin value. This is because of the nested loops and the decrementing `maximum` value. A more optimal solution would sort the coins and have a time complexity of O(N log N).
 * Space complexity: O(N) to store the values of the coins.
 */
#include <iostream>

using namespace std;

int main() {
  // total: sum of all coin values.
  // maximum: the current largest coin value being considered.
  // money: the sum of coins taken by the twin.
  // counter: the number of coins taken.
  // flag: a control variable to exit the main loop.
  int total = 0, maximum = 0, money = 0, counter = 0;
  int flag = 0;
  // coins: the total number of coins.
  int coins;
  cin >> coins;
  int value[coins];
  /**
   * @brief Read coin values and calculate the total sum.
   * @invariant This loop reads each of the `coins` values, adds it to `total`, and updates `maximum`.
   */
  for (int i = 0; i < coins; i++) {
    cin >> value[i];
    total = total + value[i];
    if (value[i] > maximum) {
      maximum = value[i];
    }
  }

  /**
   * @brief Main greedy selection loop.
   * @invariant It continues as long as the sum of money collected is not greater than half of the total.
   * The loop iterates by picking coins of the current `maximum` value.
   */
  while (flag == 0) {
    /**
     * @brief Inner loop to find all coins with the current `maximum` value.
     * @invariant Iterates through all coins to find those matching the current `maximum`.
     */
    for (int i = 0; i < coins; i++) {
      if (maximum == value[i]) {
        money = money + value[i];
        counter++;
        /**
         * @brief Check if the goal is met.
         * @post If the collected `money` is more than half of the `total`, the process stops and the result is printed.
         */
        if (money > (total / 2)) {
          cout << counter;
          flag = 1;
          break;
        }
      }
    }
    // If the flag is set, break the outer loop as well.
    if (flag == 1) {
      break;
    }
    // Decrement the `maximum` value to consider the next largest coins in the following iteration.
    maximum--;
  }
  return 0;
}
