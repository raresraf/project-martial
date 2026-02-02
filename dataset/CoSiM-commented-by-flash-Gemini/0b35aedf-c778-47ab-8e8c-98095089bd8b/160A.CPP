/**
 * @file 160A.cpp
 * @brief Solution to the "Twins" problem (Codeforces 160A).
 * This program finds the minimum number of coins needed to take such that their sum
 * is strictly greater than the sum of the remaining coins.
 * Algorithm: Greedy approach. Sorts the coins in descending order and iteratively
 * picks the largest coins until their sum exceeds the sum of the remaining coins.
 * Time Complexity: O(N log N) due to sorting, where N is the number of coins.
 * Space Complexity: O(N) for storing the coins in an array.
 */
#include <algorithm>
#include <iostream>
using namespace std;
int main() {
  int n;
  // Functional Utility: Reads the total number of coins available.
  cin >> n;
  // Initialize an array to store coin values and a variable to store the total sum of all coins.
  int arr[101], total = 0;
  /**
   * Block Logic: Reads coin values from input and calculates their total sum.
   * Pre-condition: `n` represents the count of coins to read.
   * Invariant: After the loop, `arr` contains all coin values and `total` holds their sum.
   */
  for (int i = 0; i < n; i++) {
    cin >> arr[i]; // Reads the value of each coin.
    total += arr[i]; // Accumulates the total sum of all coins.
  }
  // Initialize variables for the sum of coins taken (`coin`), count of coins taken (`cnt`),
  // and an index for iterating from the largest coin after sorting.
  int coin = 0, cnt = 0, i = n - 1;
  // Functional Utility: Sorts the coin array in ascending order to facilitate a greedy approach
  // by picking the largest coins first from the end of the array.
  sort(arr, arr + n);
  /**
   * Block Logic: Greedily selects coins from largest to smallest until the sum of selected coins
   * is strictly greater than the sum of the remaining coins.
   * Pre-condition: `arr` is sorted, `coin` and `cnt` are initialized to zero, `i` points to the largest coin.
   * Invariant: In each iteration, the largest available coin is added to `coin`, removed from `total`, and `cnt` is incremented.
   * The loop continues as long as `coin` is not greater than `total` (sum of remaining coins) and there are coins left.
   */
  while (coin <= total && i >= 0) {
    coin += arr[i]; // Add the largest available coin to the selected pile.
    total -= arr[i]; // Subtract the selected coin from the total sum of remaining coins.
    cnt++; // Increment the count of selected coins.
    i--; // Move to the next largest coin.
  }
  // Output the minimum count of coins required.
  cout << cnt;
}