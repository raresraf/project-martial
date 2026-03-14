/**
 * @67496989-76bd-416a-b3e8-ad807003f860/160A.CPP
 * @brief This program solves Codeforces problem 160A, "Twins".
 * Given 'n' coin values, the objective is to find the minimum number of coins
 * that can be taken such that their sum is strictly greater than the sum of
 * the remaining coins.
 *
 * Algorithm: Greedy approach. It calculates the total sum, sorts the coins
 * in ascending order, then iterates from the largest coins, accumulating their
 * sum until it exceeds half of the total sum.
 * Time Complexity: O(N log N) due to sorting.
 * Space Complexity: O(N) for storing coin values in an array.
 */
#include <algorithm> // For std::sort.
#include <iostream>  // For standard input/output operations (cin, cout).
using namespace std; // Uses the standard namespace.

/**
 * @brief Main function to execute the Twins problem logic.
 * Reads coin values, calculates total sum, sorts coins, and determines
 * the minimum number of coins needed to exceed half the total sum.
 * @return 0 on successful execution.
 */
int main() {
  int n; // Declares 'n', representing the number of coins.
  cin >> n; // Reads 'n' from standard input.

  // Declares an array `coins` of size `n` to store coin values.
  // `total` stores the sum of all coins.
  // `result` counts the number of coins taken.
  int coins[n], total = 0, result = 0; 
  
  /**
   * Block Logic: Reads 'n' coin values into the `coins` array and calculates their total sum.
   * Invariant: After this loop, `coins` contains all input values and `total` holds their sum.
   */
  for (int i = 0; i < n; i++) { // Loops 'n' times, once for each coin.
    cin >> coins[i]; // Reads a coin value.
    total += coins[i]; // Adds the coin value to the total sum.
  }

  // Inline: Divides `total` by 2. This represents the threshold (half the sum) that
  // the current player's collected sum must strictly exceed. Integer division is appropriate here.
  total /= 2; 
  
  // Sorts the `coins` array in ascending order.
  sort(coins, coins + n); 

  int sum = 0; // Accumulates the sum of coins taken by the current player.
  
  /**
   * Block Logic: Iterates through the sorted coins from largest to smallest, greedily taking them.
   * Accumulates their sum (`sum`) and counts the number of coins taken (`result`).
   * Stops when `sum` strictly exceeds `total` (half the original sum).
   * Invariant: `result` contains the minimum number of coins needed to exceed half the total sum.
   */
  for (int i = n - 1; i >= 0; i--) { // Loops from the last element (largest coin) downwards.
    sum += coins[i]; // Adds the current largest coin to the player's sum.
    result++;        // Increments the count of coins taken.
    // Pre-condition: Check if the current player's sum is strictly greater than half the total sum.
    if (sum > total) 
      break;         // If condition met, stop taking coins.
  }

  cout << result << endl; // Prints the minimum number of coins taken, followed by a newline.

  return 0; // Indicates successful program termination.
}