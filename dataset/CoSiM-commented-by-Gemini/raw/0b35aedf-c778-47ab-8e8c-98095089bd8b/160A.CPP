/**
 * @file 160A.CPP
 * @brief Solves the "Twins" problem using an optimal greedy algorithm.
 * @details The problem requires finding the minimum number of coins to take so that their sum is
 * strictly greater than the sum of the remaining coins. This solution implements an efficient
 * greedy strategy to achieve this.
 *
 * @algorithm Greedy Algorithm
 * 1. Read all coin values and calculate their total sum.
 * 2. Sort the coins in non-decreasing order.
 * 3. Iterate from the largest coin downwards, adding its value to a personal sum and
 *    subtracting it from the total sum.
 * 4. Count the number of coins taken.
 * 5. Stop when the personal sum becomes strictly greater than the remaining total sum.
 * 6. The final count is the minimum number of coins required.
 *
 * @complexity
 * Time complexity: O(N log N) dominated by the sorting step, where N is the number of coins.
 * Space complexity: O(N) to store the coin values in an array.
 */
#include <algorithm>
#include <iostream>
using namespace std;
int main() {
  // n: the number of coins.
  int n;
  cin >> n;
  // arr: stores the value of each coin. 101 is a safe upper bound from problem constraints.
  // total: the sum of all coin values.
  int arr[101], total = 0;
  
  /**
   * @brief Read coin values and calculate the total sum.
   * @invariant Loops `n` times to read each coin's value and add it to `total`.
   */
  for (int i = 0; i < n; i++) {
    cin >> arr[i];
    total += arr[i];
  }
  
  // coin: the sum of coins taken by the twin.
  // cnt: the number of coins taken.
  // i: loop index, starts from the last element of the sorted array.
  int coin = 0, cnt = 0, i = n - 1;
  
  // Sort the coins in ascending order.
  sort(arr, arr + n);
  
  /**
   * @brief Greedily take the largest coins.
   * @invariant In each iteration, the largest remaining coin is taken.
   * @post The loop terminates when the sum of taken coins (`coin`) is strictly greater than
   * the sum of the remaining coins (`total`).
   */
  while (coin <= total && i >= 0) {
    // Take the largest available coin.
    coin += arr[i];
    // The remaining sum decreases.
    total -= arr[i];
    // Increment the count of taken coins.
    cnt++;
    // Move to the next largest coin.
    i--;
  }
  
  // Print the minimum number of coins required.
  cout << cnt;
}
