/**
 * @6437c2f6-d9c2-4060-b4dc-d82b9ec330f3/1343C.CPP
 * @brief This program solves a problem where an array of integers needs to be processed
 * to find the maximum possible sum by selecting numbers from contiguous subarrays
 * of alternating signs. Specifically, it groups consecutive numbers of the same sign
 * and for each group, it selects the maximum absolute value from that group.
 * The overall sum is then the sum of these maximums.
 *
 * Algorithm: Greedy approach. Iterates through the array, identifying contiguous
 * subarrays of numbers with the same sign. For each such subarray, it finds the
 * maximum element and adds it to the total sum.
 * Time Complexity: O(N) where N is the number of elements in the array.
 * Space Complexity: O(N) for storing the input array (can be O(1) if processed on the fly).
 */
#include <bits/stdc++.h> // Includes many standard C++ library headers.
#include <iostream>      // Explicitly included for iostream.
using namespace std;     // Uses the standard namespace.

#define ll long long // Defines 'll' as a shorthand for long long.
#define sd long long // Defines 'sd' as a shorthand for long long (unused).

/**
 * @brief Checks if two integers have the same sign (both positive or both negative).
 * Zero is treated as positive for this function's logic.
 *
 * @param x The first integer.
 * @param y The second integer.
 * @return True if x and y have the same sign, false otherwise.
 */
bool find_(int x, int y) {
  // Block Logic: Determines if both numbers are positive or both are negative.
  // This effectively checks if their product is positive, but avoids overflow.
  if ((x > 0 && y > 0) || (x < 0 && y < 0))
    return true; // Both positive or both negative.
  else
    return false; // Signs are different.
}

/**
 * @brief Main function to process an array for maximum sum from alternating-sign groups.
 * Handles multiple test cases. For each test case, it reads an array, groups
 * numbers by sign, finds the maximum in each group, and sums these maximums.
 * @return 0 on successful execution.
 */
int main() {
  int t; // Declares 't' for the number of test cases.
  cin >> t; // Reads the number of test cases.
  /**
   * Block Logic: Processes each test case.
   * Invariant: The loop continues until all 't' test cases are processed.
   */
  while (t--) {
    int n; // Declares 'n' for the size of the array in the current test case.
    cin >> n; // Reads the array size.
    int a[n + 1]; // Declares an array 'a' to store the integers.
    /**
     * Block Logic: Reads 'n' integers into the array 'a'.
     * Invariant: After this loop, 'a' contains the input numbers for the current test case.
     */
    for (int i = 0; i < n; i++)
      cin >> a[i];

    int i = 0, j = 0; // 'i' is the main pointer, 'j' is for scanning within a group.
    ll sum = 0;       // Accumulates the total maximum sum.
    /**
     * Block Logic: Iterates through the array, identifying and processing groups of
     * consecutive numbers with the same sign.
     * Invariant: `sum` stores the cumulative maximums from each processed group.
     */
    while (i < n) {
      j = i; // Start of the current group.
      int num = a[j]; // Initialize 'num' with the first element of the group (this will hold the maximum).
      
      /**
       * Block Logic: Scans forward to find the end of the current group of same-sign numbers.
       * Simultaneously, it updates `num` to be the maximum absolute value within this group.
       * Pre-condition: `j` is within bounds and `a[j]` is the start of a group.
       * Invariant: `j` points to the last element of the group, and `num` holds the maximum
       * (by absolute value) from `a[i]` to `a[j]`.
       */
      while (j + 1 < n && find_(a[j], a[j + 1])) {
        num = max(num, a[j + 1]); // Update 'num' with the maximum element found so far in the group.
        j++; // Move to the next element.
      }
      sum += num; // Add the maximum of the current group to the total sum.
      i = j + 1;  // Move the main pointer 'i' to the beginning of the next group.
    }
    cout << sum << endl; // Prints the calculated total sum for the current test case.
  }
  return 0; // Indicates successful program termination.
}
