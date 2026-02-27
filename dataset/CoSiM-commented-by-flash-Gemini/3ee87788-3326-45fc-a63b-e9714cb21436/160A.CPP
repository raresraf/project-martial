/**
 * @file 160A.CPP
 * @brief Solution for competitive programming problem: "Twins".
 *
 * @details This program aims to find the minimum number of coins a person needs
 *          to take from a given set of coins such that the sum of their chosen
 *          coins is strictly greater than the sum of the remaining coins.
 *          Algorithm: Greedy approach - input coins are stored, total sum is calculated,
 *                     then coins are sorted in ascending order. The largest coins
 *                     are then picked iteratively until their sum is strictly
 *                     greater than the sum of the remaining coins.
 *          Time Complexity: O(N log N) due to sorting, where N is the number of coins.
 *          Space Complexity: O(N) to store the coins.
 *          Domain: Competitive Programming.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;          // Declare 'n' for the number of coins.
  cin >> n;       // Read the number of coins.
  int total_sum = 0; // Initialize total sum of all coins.
  vector<int> m;  // Declare a vector to store the coin values.
  
  // Block Logic: Read all coin values and calculate their total sum.
  // Invariant: 'm' contains all coin values, and 'total_sum' is their sum.
  for (int i = 0; i < n; i++) {
    int temp;     // Temporary variable to read each coin value.
    cin >> temp;  // Read the coin value.
    m.push_back(temp); // Add the coin to the vector.
    total_sum += temp; // Add to the total sum.
  }

  // Sort the coins in non-decreasing (ascending) order.
  sort(m.begin(), m.end());

  int sum = 0;       // Accumulates the sum of coins chosen by the current person.
  int coins = 0;     // Counts the number of coins chosen by the current person.
  // Block Logic: Greedily pick coins from largest to smallest until 'sum' is strictly greater than 'total_sum - sum'.
  // This effectively means 'sum' > 'total_sum / 2'.
  // Invariant: 'sum' is the sum of the 'coins' largest coins picked so far.
  //            'total_sum - sum' is the sum of the remaining coins.
  for (int i = n - 1; i >= 0 && sum <= total_sum - sum; i--) { // Iterate from the largest coin (end of sorted vector).
    sum += m[i];   // Add the current largest coin to the chosen sum.
    coins++;       // Increment the count of chosen coins.
  }

  cout << coins; // Output the minimum number of coins needed.
}
