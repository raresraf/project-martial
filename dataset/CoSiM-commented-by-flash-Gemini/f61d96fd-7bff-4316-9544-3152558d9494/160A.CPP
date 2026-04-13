/**
 * @file 160A.CPP
 * @brief This program solves the "Twins" problem. Given a set of coins, it determines
 * the minimum number of coins that need to be taken such that the sum of the taken
 * coins is strictly greater than the sum of the remaining coins.
 * Algorithm: Greedy approach. The coins are first sorted in ascending order. Then,
 * starting from the largest coins, they are added to a "taken sum" until this sum
 * exceeds the sum of the remaining coins.
 * Time Complexity: O(N log N) primarily due to the sorting step, where N is the number of coins.
 * Space Complexity: O(N) for storing the coin values in a vector.
 */

#include <bits/stdc++.h> // Commonly used header in competitive programming, includes many standard libraries like iostream, vector, algorithm.
using namespace std;

int main() {
  long long total_coins; // Stores the total number of coins available.
  // Reads the total number of coins.
  cin >> total_coins;

  long long sum = 0; // Stores the sum of all coin values.
  vector<long long> coin_values; // A dynamic array to store the value of each coin.
  long long temp; // Temporary variable to read coin values.

  /**
   * Block Logic: Reads the values of 'total_coins' from standard input and calculates their overall sum.
   * Each coin value is also stored in the 'coin_values' vector.
   * Invariant: After this loop, 'coin_values' contains all coin values, and 'sum' holds their total.
   */
  for (int index = 0; index < total_coins; index++) {
    cin >> temp;                  // Reads each coin's value.
    coin_values.push_back(temp);  // Adds the coin's value to the vector.
    sum += temp;                  // Adds the coin's value to the total sum.
  }

  // Sorts the 'coin_values' vector in ascending order.
  sort(coin_values.begin(), coin_values.end());

  long long counter = 0; // Counts the minimum number of coins required.
  long long out_sum = 0; // Stores the sum of coins currently taken.
  /**
   * Block Logic: Iterates through the sorted coins from largest to smallest (by traversing
   * the vector in reverse). It greedily adds coins to 'out_sum' and increments 'counter'
   * until 'out_sum' is strictly greater than the sum of the remaining coins ('sum - out_sum').
   * Invariant: 'out_sum' represents the sum of 'counter' largest coins taken so far.
   */
  for (long long out_index = coin_values.size() - 1; out_index >= 0;
       out_index--) {
    out_sum += coin_values[out_index]; // Add the current largest coin to the sum of taken coins.
    counter++;                         // Increment the count of coins taken.
    // Checks if the sum of taken coins is strictly greater than the sum of the remaining coins.
    if (out_sum > sum - out_sum) {
      cout << counter; // Prints the minimum number of coins required.
      break;           // Breaks the loop as the condition is met.
    }
  }
  // Standard program exit.
  return 0;
}