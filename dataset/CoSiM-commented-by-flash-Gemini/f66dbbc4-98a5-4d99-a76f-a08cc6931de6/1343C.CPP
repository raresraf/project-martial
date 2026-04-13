/**
 * @file 1343C.CPP
 * @brief This program solves the "Alternating Subsequence" problem. Given an
 *        array, it finds the maximum possible sum of an alternating subsequence,
 *        where positive and negative numbers must alternate.
 * Algorithm: Greedy approach. It iterates through the array, maintaining the
 *            current alternating subsequence and always picking the largest
 *            number of the current sign group.
 * Time Complexity: O(N) for each test case, where N is the size of the input array.
 * Space Complexity: O(N) for storing the input array, or O(1) if processed online.
 */
// https://codeforces.com/contest/1343/problem/C

#include <iostream>

#include <bits/stdc++.h>

using namespace std;

/**
 * @brief Determines the sign of an integer.
 * @param n The integer to check.
 * @return 1 if n > 0, -1 if n < 0, and 0 if n == 0.
 * Functional Utility: Simplifies logic for checking positive/negative alternation.
 */
int sign(int n) {
  if (n == 0)
    return 0;
  if (n > 0)
    return 1;
  return -1;
}

int main() {
  int t; // Number of test cases.
  // Reads the number of test cases.
  cin >> t;

  /**
   * Block Logic: Processes each test case.
   * Invariant: For each iteration, a new array is processed to find its maximum alternating subsequence sum.
   */
  while (t--) {
    int n; // Size of the array for the current test case.
    // Reads the size of the array.
    cin >> n;

    int a[n]; // Input array for the current test case.
    int sign_prev = 0; // Stores the sign of the previously chosen element (1 for positive, -1 for negative).
    int prev_val = 0;  // Stores the value of the previously chosen element.
    int temp;          // Temporary variable to store the sign of the current element.
    long long int ans = 0; // Accumulates the maximum sum of the alternating subsequence.

    /**
     * Block Logic: Iterates through the input array to construct the maximum alternating subsequence sum.
     * Invariant: 'ans' is the maximum alternating sum up to the current point,
     *            'sign_prev' holds the sign of the last added element,
     *            'prev_val' holds the value of the last added element.
     */
    for (int i = 0; i < n; i++) {
      cin >> a[i]; // Reads the current element of the array.

      if (i == 0) { // Special handling for the first element.
        ans += a[i]; // Adds the first element to the sum.
        sign_prev = sign(a[i]); // Records its sign.
        prev_val = a[i];         // Records its value.
      } else {
        temp = sign(a[i]); // Gets the sign of the current element.

        if (temp == sign_prev) { // If current element has the same sign as the previous.
          // Keep the current element if it's larger than the previous one of the same sign.
          ans = ans - prev_val + max(prev_val, a[i]);
          prev_val = max(prev_val, a[i]); // Update prev_val to the larger of the two.
        } else { // If current element has an alternating sign.
          ans += a[i];         // Add the current element to the sum.
          prev_val = a[i];     // Update prev_val to the current element.
          sign_prev = temp;    // Update sign_prev to the current element's sign.
        }
      }
    }
    // Prints the maximum sum of the alternating subsequence for the current test case.
    cout << ans << endl;
  }
}