/**
 * @file 160A.CPP
 * @brief Solution to the "Twins" problem from Codeforces (Problem 160A).
 * Algorithm: Greedy approach. Sorts the coins in descending order and iteratively
 *            selects the largest coins until their sum exceeds half of the total sum.
 * Time Complexity: O(N log N) due to sorting, where N is the number of coins.
 * Space Complexity: O(N) for storing the coins.
 * Domain: Competitive Programming, Greedy Algorithms.
 */
#include <bits/stdc++.h>

using namespace std;

/**
 * @brief Main function to solve the Twins problem.
 * Functional Utility: Reads the number of coins and their values, then determines
 *                     the minimum number of coins required to achieve a sum strictly
 *                     greater than the remaining coins.
 */
int main() {
  // Variables:
  // n: Total number of coins.
  // i: Loop counter.
  // sum: Total sum of all coins.
  // sum1: Current sum of coins taken by one twin.
  // e: Count of coins taken by one twin.
  // sum_com: Cumulative sum of coins taken (used to compare against `sum`).
  int n, i, sum = 0, sum1 = 0, e = 0, sum_com = 0;
  cin >> n;
  int arr[n];
  // Block Logic: Reads coin values from input and calculates the total sum of all coins.
  for (i = 0; i < n; i++) {
    cin >> arr[i];
    sum = sum + arr[i];
  }
  // Block Logic: Sorts the coins in ascending order.
  // Pre-condition: `arr` contains all coin values.
  // Post-condition: `arr` is sorted in ascending order.
  sort(arr, arr + n);

  // Block Logic: Iteratively selects coins from largest to smallest (due to reverse iteration after sort)
  //              to find the minimum number of coins whose sum is strictly greater than the sum of the remaining coins.
  // Invariant: At the start of each iteration, `sum_com` holds the sum of coins already taken,
  //            and `sum` holds the sum of the remaining coins.
  for (i = n - 1; i >= 0; i--) {
    sum1 = sum1 + arr[i]; // Add the current largest coin to the twin's sum.
    sum = sum - arr[i];   // Subtract the taken coin from the total sum of remaining coins.
    e++;                  // Increment the count of coins taken.
    // Block Logic: Compares the sum of coins taken (`sum1` after being added to `sum_com`)
    //              with the sum of remaining coins (`sum`).
    if (sum1 + sum_com > sum) // Check if current twin's cumulative sum is strictly greater than the other twin's sum.
      break;
  }
  // Block Logic: Outputs the minimum number of coins (`e`) found.
  cout << e;

  return 0;
}
