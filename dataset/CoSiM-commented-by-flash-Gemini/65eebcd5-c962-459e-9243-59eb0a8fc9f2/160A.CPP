/**
 * @65eebcd5-c962-459e-9243-59eb0a8fc9f2/160A.CPP
 * @brief This program solves Codeforces problem 160A, "Twins".
 * Given 'n' coin values, the objective is to find the minimum number of coins
 * that can be taken such that their sum is strictly greater than the sum of
 * the remaining coins.
 *
 * Algorithm: Greedy approach. It calculates the total sum, sorts the coins
 * in descending order, and then iteratively takes the largest coins until
 * their sum exceeds the sum of the remaining coins.
 * Time Complexity: O(N log N) due to sorting.
 * Space Complexity: O(N) for storing coin values in a vector.
 */
#include <algorithm> // For std::sort and std::reverse.
#include <iostream>  // For standard input/output operations (cin, cout).
#include <vector>    // For using std::vector.

using namespace std; // Uses the standard namespace.

/**
 * @brief Main function to execute the Twins problem logic.
 * Reads coin values, calculates total sum, sorts coins, and determines
 * the minimum number of coins needed to exceed half the total sum.
 * @return 0 on successful execution.
 */
int main() {
  int n = 0, a = 0; // Declares 'n' for the number of coins, 'a' for reading individual coin values.
  int m = 0;        // 'm' will store the total sum of all coins.
  int sum = 0;      // 'sum' is declared but not used in this specific implementation; `temp_sum` serves its purpose.
  int k = 0;        // 'k' counts the number of coins taken.
  int temp_sum = 0; // 'temp_sum' accumulates the sum of coins taken by the current player.

  cin >> n; // Reads 'n' (number of coins) from standard input.
  vector<int> v; // Declares a vector to store the coin values.

  /**
   * Block Logic: Reads 'n' coin values into the vector `v` and calculates their total sum `m`.
   * Invariant: After this loop, `v` contains all coin values and `m` holds their total sum.
   */
  for (int i = 0; i < n; i++) {
    cin >> a;          // Reads each coin value.
    v.push_back(a);    // Adds the coin to the vector.
    m += a;            // Adds the coin value to the total sum 'm'.
  }

  // Sorts the vector `v` in ascending order.
  sort(v.begin(), v.end()); 
  // Reverses the sorted vector `v`, so coins are now in descending order.
  reverse(v.begin(), v.end()); 

  /**
   * Block Logic: Iterates through the coins (from largest to smallest) and takes
   * them until the sum of taken coins (`temp_sum`) is strictly greater than the
   * sum of the remaining coins (`m`).
   * Invariant: `k` counts the minimum number of coins needed to satisfy the condition.
   */
  for (int i = 0; i < n; i++) { // Loops through the coins in descending order.
    temp_sum += v[i]; // Adds the current largest coin to `temp_sum`.
    k++;              // Increments the count of coins taken.
    m = m - v[i];     // Subtracts the current coin from the remaining sum (`m`).
    if (temp_sum > m) // Pre-condition: Check if the current player's sum is strictly greater than the remaining sum.
      break;          // If condition met, stop taking coins.
  }
  cout << k; // Prints the minimum number of coins taken.
  return 0;  // Indicates successful program termination.
}