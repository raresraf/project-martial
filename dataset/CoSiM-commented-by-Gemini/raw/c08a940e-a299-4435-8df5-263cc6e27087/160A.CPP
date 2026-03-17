/**
 * @file 160A.CPP
 * @brief Solution to a competitive programming problem, likely Codeforces 160A "Twins".
 *
 * @details
 * This program solves a greedy problem where the goal is to find the minimum
 * number of coins to take from a set so that their sum is strictly greater than
 * the sum of the remaining coins.
 *
 * @algorithm
 * The optimal strategy is a greedy one: always take the largest available coin.
 * 1. Read all coin values into a vector and calculate the total sum.
 * 2. Sort the vector of coins in non-decreasing (ascending) order.
 * 3. Iteratively take the largest coin from the end of the sorted vector.
 * 4. Add the coin's value to a personal sum and increment a coin counter.
 * 5. After each coin is taken, check if the personal sum has become strictly
 *    greater than the sum of the remaining coins.
 * 6. Stop as soon as this condition is met. The number of coins taken is the answer.
 *
 * @complexity
 * - Time Complexity: O(N log N), where N is the number of coins, dominated by the
 *   `std::sort` operation.
 * - Space Complexity: O(N), to store the coin values in a vector.
 */
#include <algorithm>
#include <bits/stdc++.h>

using namespace std;

/**
 * @brief Main function and entry point of the program.
 */
int main() {
  int n; // Number of coins.
  int x; // Temporary variable for reading coin values.
  int count = 0; // The number of coins taken.
  int sum = 0; // The total sum of all coins.
  int sum1 = 0; // The sum of coins taken.

  cin >> n;

  vector<int> a;

  // Read coin values into the vector and calculate the total sum.
  for (int i = 0; i < n; i++) {
    cin >> x;
    sum = sum + x;
    a.push_back(x);
  }

  // Sort the coins in non-decreasing (ascending) order.
  sort(a.begin(), a.end());

  /**
   * @brief Greedily take the largest coins until the sum is sufficient.
   * This loop simulates taking coins one by one from largest to smallest.
   */
  while (1) {
    // Take the largest available coin (from the end of the sorted vector).
    sum1 = sum1 + a[n - 1];

    // Calculate the sum of the remaining coins.
    int s = sum - sum1;

    count++;

    // Check if the sum of taken coins is strictly greater than the remaining sum.
    if (s < sum1) {
      break; // If so, we have taken enough coins.
    }
    
    // "Remove" the coin just taken by decrementing the effective size of the array.
    n--;
  }

  // Output the minimum number of coins required.
  cout << count;
}
