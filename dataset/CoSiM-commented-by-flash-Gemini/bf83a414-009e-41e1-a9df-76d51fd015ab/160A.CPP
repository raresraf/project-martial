/**
 * @file 160A.CPP
 * @brief This program solves the "Twins" problem. Given a collection of coins,
 * it determines the minimum number of coins a person needs to take such that
 * the sum of these coins is strictly greater than the sum of the remaining coins.
 * Algorithm: Bubble sort to arrange coins in descending order, followed by a greedy
 * approach to pick coins from largest to smallest until the sum condition is met.
 * Time Complexity: O(N^2) due to the bubble sort, where N is the number of coins.
 * Space Complexity: O(N) for storing the coin values in an array.
 */
#include <iostream> // Required for input/output operations (std::cin, std::cout).

int main() {
  int numCoins = 0; // Stores the total number of coins available.
  int coins[100];   // A C-style array to store the value of each coin (max 100 coins).
  int twinsSum = 0; // Initially stores the total sum of all coins. Later, stores the sum of remaining coins.
  int yourSum = 0;  // Stores the sum of coins taken by "you".
  int answer = 0;   // Stores the minimum number of coins taken.

  // Reads the total number of coins.
  std::cin >> numCoins;
  /**
   * Block Logic: Reads the values of 'numCoins' from standard input and calculates their total sum.
   * Invariant: After this loop, 'coins' array contains all coin values and 'twinsSum' is their total sum.
   */
  for (int i = 0; i < numCoins; i++) {
    std::cin >> coins[i];      // Reads each coin's value.
    twinsSum += coins[i];       // Adds the coin's value to the total sum.
  }

  /**
   * Block Logic: Sorts the 'coins' array in descending order using a bubble sort-like mechanism.
   * The `i = -1` statement effectively restarts the outer loop after each swap to ensure
   * that larger elements bubble up to the beginning of the array.
   * Invariant: After this loop, 'coins' array is sorted from largest to smallest.
   */
  for (int i = 0; i < numCoins - 1; i++) {
    int temp; // Temporary variable for swapping.
    // Compares adjacent elements and swaps them if they are in the wrong order (for descending sort).
    if (coins[i] < coins[i + 1]) {
      temp = coins[i];
      coins[i] = coins[i + 1];
      coins[i + 1] = temp;
      i = -1; // Restart the outer loop to ensure proper bubbling of largest elements.
    }
  }

  /**
   * Block Logic: Greedily picks coins from the largest to smallest until 'yourSum'
   * (sum of taken coins) is strictly greater than 'twinsSum' (sum of remaining coins).
   * Invariant: 'yourSum' accumulates the sum of the 'answer' largest coins, and
   * 'twinsSum' is the sum of all other coins.
   */
  for (int i = 0; yourSum <= twinsSum; i++) { // Loop continues as long as 'yourSum' is not greater than 'twinsSum'.
    yourSum += coins[i];  // Add the current largest coin to 'yourSum'.
    twinsSum -= coins[i]; // Subtract the taken coin from 'twinsSum' (sum of remaining coins).
    answer++;             // Increment the count of coins taken.
  }
  // Prints the minimum number of coins required.
  std::cout << answer << "\n";
  // Standard program exit.
  return 0;
}
