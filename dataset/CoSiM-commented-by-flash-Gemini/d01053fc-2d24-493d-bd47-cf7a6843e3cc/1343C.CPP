/**
 * @file 1343C.CPP
 * @brief This program solves a problem by finding the maximum sum of a subsequence
 * where numbers alternate in sign (positive, negative, positive, negative, etc.).
 * From each contiguous block of same-sign numbers, only the maximum absolute value
 * of that sign is selected.
 * Algorithm: Iterative greedy selection. The array is traversed, and from each
 * continuous segment of either all positive or all negative numbers, the maximum
 * value is picked and added to the total sum.
 * Time Complexity: O(N) per test case, where N is the size of the array, due to a single pass.
 * Space Complexity: O(N) for storing the array.
 */
#include <bits/stdc++.h> // Commonly used header in competitive programming for convenience, includes many standard libraries.
using namespace std;     // Brings all names from the std namespace into the current scope.

#define endl "\n" // Defines a macro 'endl' to print a newline character efficiently.

int main() {
  // Optimizes C++ standard streams for faster input/output operations.
  ios_base ::sync_with_stdio(false);
  cin.tie(0);  // Unties cin from cout.
  cout.tie(0); // Unties cout.

  int t, n; // 't' for number of test cases, 'n' for array size in each test case.
  // Reads the number of test cases.
  cin >> t;
  /**
   * Block Logic: Processes each of the 't' test cases.
   * Each iteration reads an array of numbers and computes the maximum alternating sum.
   * Invariant: `t` decrements until all test cases are processed.
   */
  while (t--) {
    // Reads the size of the array for the current test case.
    cin >> n;
    // Declares a Variable Length Array (VLA) to store the array elements.
    // Note: VLAs are a C99 feature and a GNU extension in C++, not standard C++.
    int a[n];
    /**
     * Block Logic: Reads 'n' integer elements into the array 'a'.
     */
    for (int i = 0; i < n; i++) {
      cin >> a[i]; // Reads each element.
    }

    long long sum = 0; // Accumulates the maximum alternating sum.
    int i = 0;         // Current index for traversing the array 'a'.
    /**
     * Block Logic: Iterates through the array 'a' to find alternating segments of numbers.
     * It processes segments of negative numbers, then positive numbers, and so on.
     * From each segment, it picks the maximum value.
     * Invariant: 'i' progresses through the array, and 'sum' accumulates the maximums of alternating segments.
     */
    while (i < n) {
      int maxm = INT_MIN; // Stores the maximum value found in the current segment.

      /**
       * Block Logic: Finds the maximum negative number in a continuous segment of negative numbers.
       * Continues as long as the current index is within bounds and the number is negative.
       */
      while (i < n && a[i] < 0) {
        maxm = max(maxm, a[i]); // Updates 'maxm' if a larger negative number is found.
        i++;                    // Moves to the next element.
      }
      if (maxm != INT_MIN) { // If a negative segment was found.
        sum += maxm;         // Add the maximum negative number to the total sum.
      }

      maxm = INT_MIN; // Resets 'maxm' for the next segment.
      /**
       * Block Logic: Finds the maximum positive number in a continuous segment of positive numbers.
       * Continues as long as the current index is within bounds and the number is positive.
       */
      while (i < n && a[i] > 0) {
        maxm = max(maxm, a[i]); // Updates 'maxm' if a larger positive number is found.
        i++;                    // Moves to the next element.
      }
      if (maxm != INT_MIN) { // If a positive segment was found.
        sum += maxm;         // Add the maximum positive number to the total sum.
      }
    }
    // Prints the total maximum alternating sum for the current test case.
    cout << sum << endl;
  }

  // Standard program exit.
  return 0;
}