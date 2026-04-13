/**
 * @file 160A.CPP
 * @brief This program solves the "Twins" problem. Given a collection of coins,
 * it determines the minimum number of coins a person needs to take such that
 * the sum of these coins is strictly greater than the sum of the remaining coins.
 * Algorithm: Greedy approach. Sorts the coins in descending order and iteratively
 * takes the largest coins until their sum exceeds the sum of the remaining coins.
 * Time Complexity: O(N log N) primarily due to the sorting step, where N is the number of coins.
 * Space Complexity: O(N) for storing the coin values in a vector.
 */
#include <algorithm> // Required for std::sort.
#include <iostream>  // Required for input/output operations (cin, cout).
#include <vector>    // Required for std::vector.

using namespace std;

int main() {
  int num_coins; // Stores the total number of coins available.
  int total_sum = 0; // Stores the sum of all coins.
  // Reads the total number of coins.
  cin >> num_coins;
  // Creates a vector of integers to store the value of each coin, initialized to `num_coins` size.
  vector<int> coins(num_coins);
  /**
   * Block Logic: Reads the values of 'num_coins' from standard input and calculates their total sum.
   * Invariant: After this loop, 'coins' vector contains all coin values and 'total_sum' is their sum.
   */
  for (size_t i = 0; i < num_coins; ++i) {
    cin >> coins[i];      // Reads each coin's value.
    total_sum += coins[i]; // Adds the coin's value to the total sum.
  }
  // Sorts the 'coins' vector in ascending order.
  sort(begin(coins), end(coins));

  int min_coins_required = 0; // Counts the minimum number of coins taken.
  int sum = 0;                // Stores the sum of coins currently taken.
  /**
   * Block Logic: Iterates through the sorted coins (from largest to smallest)
   * to find the minimum number of coins whose sum is greater than the sum of remaining coins.
   * It takes coins one by one, starting from the largest, until the condition is met.
   * Invariant: 'sum' is the sum of 'min_coins_required' largest coins.
   */
  for (int i = num_coins - 1; i >= 0; --i) {
    sum += coins[i];            // Adds the current largest coin to the 'sum'.
    ++min_coins_required;       // Increments the count of coins taken.
    // Checks if the sum of taken coins is strictly greater than the sum of the remaining coins.
    if (sum > total_sum - sum) {
      break; // Breaks the loop as the condition is met.
    }
  }
  // Prints the minimum number of coins required to satisfy the condition.
  cout << min_coins_required << endl;
  // Standard program exit.
  return 0;
}
