/**
 * @file 160A.CPP
 * @brief Solves the "Twins" problem from Codeforces (Problem 160A).
 * This program finds the minimum number of coins whose sum is strictly greater
 * than the sum of the remaining coins.
 *
 * Algorithm: Greedy approach. Sort the coins in descending order and iteratively
 * pick the largest coins until their sum exceeds half of the total sum.
 * Time Complexity: O(N log N) due to sorting, where N is the number of coins.
 * Space Complexity: O(N) to store the coins.
 */

#include <algorithm> // Required for std::sort
#include <cstdio>    // Required for standard input/output operations (e.g., printf, scanf)
#include <cstring>   // Required for string manipulation functions (not directly used but common)
#include <iostream>  // Required for standard input/output streams (cin, cout)
#include <vector>    // Required for std::vector container
using namespace std;
int main() {
  /**
   * @brief Main function to solve the Twins problem.
   * Reads the number of coins, their values, sorts them, and applies a greedy strategy
   * to find the minimum number of coins meeting the criteria.
   */
  int n;
  vector<int> v;
  // Block Logic: Read the total number of coins (n) and their individual values.
  // Simultaneously calculate the total sum of all coins.
  cin >> n;
  long sum = 0;
  for (int i = 0; i < n; i++) {
    int x;
    cin >> x;
    sum += x;
    v.push_back(x);
  }
  // Block Logic: Sort the vector of coins in ascending order.
  // This is a prerequisite for the greedy approach to pick the largest coins first.
  sort(v.begin(), v.end());
  
  long sum2 = 0; // @brief Accumulator for the sum of selected coins.
  long count = 0; // @brief Counter for the number of selected coins.
  
  // Block Logic: Implement the greedy strategy by iterating from the largest coins.
  // Pre-condition: The vector 'v' is sorted in ascending order.
  // Invariant: In each iteration, the largest available coin is added to 'sum2'
  //            and 'count' is incremented.
  // Post-condition: 'sum2' holds the sum of the selected coins, and 'count' holds
  //                 the number of coins needed for 'sum2' to be strictly greater
  //                 than half of the total sum.
  for (int i = n - 1; i >= 0; i--) {
    sum2 += v[i];
    count++;
    if (sum2 >= sum / 2 + 1)
      break;
  }
  // Block Logic: Output the minimum number of coins required.
  cout << count << "\n";
}