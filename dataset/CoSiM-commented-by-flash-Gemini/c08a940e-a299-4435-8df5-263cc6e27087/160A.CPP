/**
 * @file 160A.CPP
 * @brief This program solves the "Twins" problem (Codeforces 160A).
 *   It determines the minimum number of coins needed to take such that the sum of these coins
 *   is strictly greater than the sum of the remaining coins.
 *   Problem description: https://codeforces.com/problemset/problem/160/A
 * * Algorithm: Greedy approach with sorting.
 *   1. Calculate the total sum of all coins.
 *   2. Sort the coins in ascending order.
 *   3. Iteratively take the largest coins (from the end of the sorted array) until the sum of taken coins
 *      is strictly greater than the sum of the remaining coins.
 * Time Complexity: O(N log N), dominated by the sorting step. Input reading and
 *   the greedy selection loop are O(N).
 * Space Complexity: O(N) for storing the vector of coin values.
 */
#include <algorithm> // Required for `std::sort`.
#include <bits/stdc++.h> // Includes other standard C++ libraries.

using namespace std;

/**
 * @brief Main function to solve the Twins problem.
 * Functional Utility: Reads coin values, calculates total sum, sorts coins,
 *                     and greedily selects coins to achieve a sum greater than remaining.
 */
int main() {
  int n; // `n`: Stores the number of coins.
  int x; // `x`: Temporary variable for reading each coin value.
  int count = 0; // `count`: Counts the number of coins taken by the "twin".
  int sum = 0;   // `sum`: Stores the total value of all coins.
  int sum1 = 0;  // `sum1`: Stores the sum of coins taken by the "twin".

  // Block Logic: Reads the total number of coins, `n`.
  cin >> n;

  // Functional Utility: Creates a vector of integers to store coin values.
  vector<int> a;

  /**
   * Block Logic: Reads `n` coin values and calculates their total sum.
   * Pre-condition: `n` is a positive integer.
   * Invariant: `a` is populated with coin values, and `sum` holds their total.
   */
  for (int i = 0; i < n; i++) {
    cin >> x;        // Reads current coin value.
    sum = sum + x;   // Adds to total sum.
    a.push_back(x);  // Adds coin to vector.
  }

  // Functional Utility: Sorts the vector `a` in ascending order.
  // This is crucial for the greedy strategy, as coins will be taken from the end (largest first).
  sort(a.begin(), a.end());

  // Commented-out block, likely for debugging or an alternative way to calculate the sum.
  //    for(int i=0; i<n; i++)
  //    {
  //        sum=sum+a[i];
  //    }

  /**
   * Block Logic: Greedily selects coins from the largest (`a[n-1]`) until their sum (`sum1`)
   *              is strictly greater than the sum of remaining coins (`s`).
   * Pre-condition: `a` is sorted, `sum` holds the total value.
   * Invariant: `sum1` increases, `s` decreases, `count` increases, `n` decreases,
   *            until `sum1 > s`.
   */
  while (1) { // Infinite loop that will break when the condition is met.
    sum1 = sum1 + a[n - 1]; // Adds the largest remaining coin to `sum1`.

    int s_remaining = sum - sum1; // Calculates the sum of remaining coins.

    count++; // Increments the count of coins taken.

    // Conditional Logic: Checks if the sum of taken coins (`sum1`) is strictly greater
    //                    than the sum of remaining coins (`s_remaining`).
    if (s_remaining < sum1) {
      break; // Exit loop if condition is met.
    }

    n--; // Decrements `n` to effectively move to the next largest coin in the sorted array.
  }

  // Block Logic: Prints the minimum number of coins needed.
  cout << count;
}
