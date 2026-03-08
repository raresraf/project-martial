/**
 * @file 160A.CPP
 * @brief Solution for the "Twins" problem on Codeforces.
 *
 * @details
 * This program finds the minimum number of coins one needs to take from a set
 * so that the sum of the taken coins is strictly greater than the sum of the
 * remaining coins. The solution employs a greedy algorithm.
 *
 * @section
 * Algorithm: Greedy approach.
 * 1. Calculate the total sum of all coins.
 * 2. Sort the coins in descending order.
 * 3. Iterate through the sorted coins, taking the largest ones first, and
 *    accumulate their value. In parallel, subtract their value from the total sum.
 * 4. Stop when the accumulated sum is strictly greater than the remaining sum.
 * The number of coins taken at that point is the answer.
 *
 * Time Complexity: O(N log N) dominated by the sorting step.
 * Space Complexity: O(N) to store the coin values.
 */
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

using namespace std;

int main() {
  int n;
  cin >> n;

  vector<int> coins(n);

  for (int i = 0; i < n; i++) {
    cin >> coins[i];
  }

  // Your sum starts at 0.
  int my_sum = 0;
  // The twin initially has all the coins. `accumulate` efficiently sums the vector.
  int twin_sum = accumulate(coins.begin(), coins.end(), 0);
  ;

  // Sort the coins in descending order. This is the core of the greedy strategy.
  sort(coins.rbegin(), coins.rend());

  // 'num_of_coins' will count the minimum number of coins needed.
  int num_of_coins{};
  /**
   * @brief This loop implements the greedy selection process.
   *
   * Logic: Take the largest available coin, update the sums for yourself and
   * your twin, and check if your sum is now strictly greater.
   */
  for (int i = 0; i < n; i++) {
    // taking a coin with the highest value, and subtract the value from my_twin
    my_sum += coins[i];
    twin_sum -= coins[i];
    num_of_coins++;

    // Pre-condition: Check if the goal has been reached.
    if (my_sum > twin_sum) {
      // Invariant: The minimum number of coins has been found, so exit the loop.
      break;
    }
  }

  // Output the minimum number of coins required.
  cout << num_of_coins << endl;
}
