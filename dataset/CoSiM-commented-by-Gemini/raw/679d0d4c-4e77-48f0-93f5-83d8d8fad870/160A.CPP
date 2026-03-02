#include <bits/stdc++.h>
#include <vector>
using namespace std;

/**
 * @brief Custom comparator function to sort integers in descending order.
 * @param x The first integer.
 * @param y The second integer.
 * @return True if x is greater than y, false otherwise.
 */
bool chg(int x, int y) { return x > y; }

/**
 * @file 160A.CPP
 * @brief Solution to the "Twins" problem from Codeforces (Problem 160A).
 * @details This program finds the minimum number of coins a person must take so that
 * their total coin value is strictly greater than the total value of the remaining coins.
 *
 * Algorithm:
 * The problem is solved using a greedy strategy. To achieve the largest sum with the fewest
 * coins, one must pick the coins with the highest values first.
 * 1. The coin values are read into a vector.
 * 2. The vector is sorted in descending order.
 * 3. The total sum of all coins is calculated.
 * 4. The program iterates through the sorted (largest to smallest) coins, adding each coin's
 *    value to a `sum` for the person taking them, and subtracting it from the `total` sum.
 * 5. The loop stops as soon as the person's `sum` becomes strictly greater than the
 *    remaining `total`. The number of coins taken is then the answer.
 *
 * Time Complexity: O(T log T), where T is the number of coins, dominated by the sort operation.
 * Space Complexity: O(T) to store the coin values in the vector.
 */
int main() {
  vector<int> vect;
  int t, n, sum = 0, i, j;
  // Read the number of coins.
  cin >> t;
  // Read each coin's value into the vector.
  while (t--) {
    cin >> n;
    vect.push_back(n);
  }
  // Sort the coins in descending order to apply the greedy strategy.
  sort(vect.begin(), vect.end(), chg);
  // Calculate the total value of all coins.
  int p = accumulate(vect.begin(), vect.end(), 0);

  /**
   * @brief Main loop to find the minimum number of coins.
   * Iterates through the coins from largest to smallest.
   */
  for (i = 0; i < vect.size(); i++) {
    // Add the current largest coin to the twin's sum.
    sum += vect[i];
    // Subtract the same from the total remaining sum.
    p = p - vect[i];
    // Check if the twin's sum is now greater than the other twin's sum.
    if (sum > p) {
      cout << i + 1 << endl;
      return 0;
    }
  }

  return 0;
}
