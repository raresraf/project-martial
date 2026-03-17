/**
 * @file 160A.CPP
 * @brief Solution to a competitive programming problem, likely Codeforces 160A "Twins".
 *
 * @details
 * This program solves a classic greedy problem. Given a set of coins, the goal is
 * to find the minimum number of coins one needs to take so that their sum is
 * strictly greater than the sum of the remaining coins.
 *
 * @algorithm
 * The optimal strategy is a greedy one:
 * 1. Calculate the total sum of all coin values.
 * 2. Sort the coins in descending order. The implementation here uses a bubble-sort-like
 *    method that repeatedly passes through the array.
 * 3. Iterate through the sorted coins from largest to smallest, taking one coin at a time.
 * 4. In each step, update the personal sum and the total sum, and count the number
 *    of coins taken.
 * 5. Stop as soon as the personal sum becomes strictly greater than the remaining sum.
 * 6. The number of coins taken is the answer.
 *
 * @complexity
 * - Time Complexity: O(N^2), where N is the number of coins. This is dominated by the
 *   inefficient bubble-sort implementation used. An O(N log N) solution could be
 *   achieved by using `std::sort`.
 * - Space Complexity: O(N) to store the coin values in an array.
 */
#include <iostream>

/**
 * @brief Main function and entry point of the program.
 */
int main() {
  int numCoins = 0;
  int coins[100]; // Array to store the value of each coin.
  int twinsSum = 0; // The total sum of all coins.
  int yourSum = 0; // The sum of coins taken.
  int answer = 0; // The number of coins taken.
  
  std::cin >> numCoins;
  
  // Read all coin values and calculate the total sum.
  for (int i = 0; i < numCoins; i++) {
    std::cin >> coins[i];
    twinsSum += coins[i];
  }

  /**
   * @brief Sort the coins in descending order.
   * This is an inefficient but functional implementation of a bubble sort.
   * When a swap is made, the loop is reset by setting `i = -1` to ensure
   * another pass is made, guaranteeing the largest elements "bubble up".
   */
  for (int i = 0; i < numCoins - 1; i++) {
    int temp;
    if (coins[i] < coins[i + 1]) {
      temp = coins[i];
      coins[i] = coins[i + 1];
      coins[i + 1] = temp;
      i = -1; // Reset loop to re-scan from the beginning after a swap.
    }
  }

  /**
   * @brief Greedily take the largest coins until the sum is sufficient.
   * The loop continues as long as `yourSum` is not strictly greater than
   * the remaining sum (`twinsSum`).
   */
  for (int i = 0; twinsSum >= yourSum; i++) {
    yourSum += coins[i];
    twinsSum -= coins[i];
    answer++;
  }
  
  std::cout << answer << "
";
}
