/**
 * @file 160A.CPP
 * @brief Solution to the "Twins" problem.
 *
 * @details This program aims to find the minimum number of coins a twin needs to take from a set
 * so that the sum of their coins is strictly greater than the sum of the coins remaining.
 * This is a classic greedy algorithm problem.
 *
 * @section
 * Algorithm:
 * 1. Calculate the total sum of all coin values.
 * 2. Sort the coins in non-decreasing order.
 * 3. Greedily take the largest coins one by one, summing their values.
 * 4. Stop when the sum of the taken coins is strictly greater than half of the total sum.
 * 5. The number of coins taken is the answer.
 *
 * Time Complexity: O(N log N), dominated by the sorting step.
 * Space Complexity: O(N), to store the coin values in a vector.
 *
 * @note The code is structured to handle multiple test cases in a single run, but uses a global
 * vector `arr` which is not cleared between cases, potentially leading to incorrect results
 * for subsequent cases.
 */
#include <bits/stdc++.h>
#include <iostream>
#include <math.h>

using namespace std;
vector<int> arr;
int main() {
  int n;
  int total = 0;
  int val;
  int takenTotal;
  /**
   * @brief This outer loop is structured to read and process multiple test cases.
   */
  while (cin >> n) {
    total = 0;
    /**
     * @brief This loop reads 'n' coin values, calculates their total sum,
     * and appends them to a global vector.
     */
    for (int i = 0; i < n; i++) {
      cin >> val;
      arr.push_back(val);
      total = total + val;
    }
    int i = arr.size();
    // Functional Utility: Sorts the array to enable the greedy approach of picking the largest coins first.
    sort(arr.begin(), arr.begin() + i);
    takenTotal = 0;
    /**
     * @brief This loop implements the greedy strategy.
     * Logic: It iterates backward from the end of the sorted array (largest coins),
     * adding them to `takenTotal` until the sum of taken coins is no longer less than or
     * equal to the sum of the remaining coins.
     */
    do {
      takenTotal = takenTotal + arr[--i];
    } while ((total - takenTotal) >= takenTotal && (i - 1 > -1));
    
    // The number of coins taken is the difference between the original size and the final position of 'i'.
    cout << arr.size() - i << endl;
  }
}
