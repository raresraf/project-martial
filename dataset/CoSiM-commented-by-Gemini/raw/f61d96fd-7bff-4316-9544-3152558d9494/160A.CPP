/**
 * @file 160A.CPP
 * @brief Solves the "Twins" problem from Codeforces (Problem 160A).
 * @details The objective is to find the minimum number of coins one needs to take
 * so that the sum of their values is strictly greater than the sum of the values
 * of the remaining coins.
 *
 * Algorithm: Greedy approach.
 * 1. Read all coin values and calculate their total sum.
 * 2. Sort the coins in non-decreasing order.
 * 3. Greedily take the coins with the largest values one by one.
 * 4. After taking each coin, check if the sum of the taken coins is strictly
 *    greater than the sum of the remaining coins.
 * 5. The number of coins taken when this condition is first met is the answer.
 *
 * Time Complexity: O(N log N) dominated by the sorting step.
 * Space Complexity: O(N) to store the coin values.
 */
// CodeForces
// Problem 160A [Twins]

#include <bits/stdc++.h>
using namespace std;

int main() {
  long long total_coins;
  cin >> total_coins;

  long long sum = 0;
  vector<long long> coin_values;
  long long temp;
  /**
   * @brief Reads coin values from input and calculates the total sum.
   */
  for (int index = 0; index < total_coins; index++) {
    cin >> temp;
    coin_values.push_back(temp);
    sum += temp;
  }
  // Sorts coins to enable the greedy selection of the largest values.
  sort(coin_values.begin(), coin_values.end());

  long long counter = 1, out_sum = 0;
  /**
   * @brief Greedily takes the largest coins and checks the condition.
   * Iterates from the largest coin downwards, adding its value to `out_sum`.
   * It checks if the current sum of taken coins (`out_sum`) is greater than
   * the sum of the remaining coins (`sum - out_sum`).
   */
  for (long long out_index = coin_values.size() - 1; out_index >= 0;
       out_index--) {
    out_sum += coin_values[out_index];
    if (out_sum > sum - out_sum) {
      cout << counter;
      break;
    } else {
      counter++;
    }
  }
  return 0;
}