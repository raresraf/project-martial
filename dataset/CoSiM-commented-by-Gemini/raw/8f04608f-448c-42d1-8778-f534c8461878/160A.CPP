/**
 * @file 160A.CPP
 * @brief Solution for the "Twins" problem (e.g., Codeforces 160A).
 * @details This program implements a greedy algorithm to solve the following problem:
 * Given a set of coins, find the minimum number of coins one needs to take so that
 * the sum of their values is strictly greater than the sum of the remaining coins.
 * The optimal strategy is to always take the coins with the highest value first.
 *
 * @algorithm Greedy Algorithm
 * @complexity
 * - Time: O(N log N), where N is the number of coins. The complexity is dominated
 *   by the sorting operation.
 * - Space: O(N) to store the values of the coins in an array.
 */
#include <algorithm>
#include <iostream>
using namespace std;

int main() {
  // t: the number of coins.
  // num: an array to store the value of each coin.
  int t, num[101];

  // The loop is structured to handle multiple test cases until End-Of-File.
  while (cin >> t && t != EOF) {
    // sum: the total value of all coins.
    // total: the sum of values of coins taken.
    // Ans: the count of coins taken.
    int sum = 0, total = 0, Ans = 0;

    /**
     * @block Input and Summation
     * @logic Reads each coin's value and calculates the total sum of all coins.
     */
    for (int i = 0; i < t; i++) {
      cin >> num[i];
      sum += num[i];
    }

    // Sort the coins in non-decreasing order to easily access the largest ones.
    sort(num, num + t);

    /**
     * @block Greedy Selection
     * @logic Iterates backwards from the largest coin, adding its value to `total`
     * until the `total` is strictly greater than half of the total `sum`.
     */
    for (int i = t - 1; i >= 0 && total <= sum / 2; i--) {
      // Take the largest available coin.
      total += num[i];
      // Increment the count of coins taken.
      Ans++;
    }

    // Output the minimum number of coins required.
    cout << Ans << endl;
  }
  return 0;
}
