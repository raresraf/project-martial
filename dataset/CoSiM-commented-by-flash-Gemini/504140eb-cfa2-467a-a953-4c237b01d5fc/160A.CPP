/**
 * @file 160A.CPP
 * @brief Implements a greedy algorithm to find the minimum number of elements
 *        whose sum is strictly greater than the sum of the remaining elements.
 *        This is a common problem in competitive programming, often referred to
 *        as "Partitioning Coins" or "Twin Problem".
 *
 * Algorithm: Greedy approach. Sorts the array in descending order and iteratively
 *            selects the largest elements, maintaining two sums: one for selected
 *            elements and one for remaining elements.
 * Time Complexity: O(N log N) due to sorting, where N is the number of elements.
 * Space Complexity: O(N) for storing the array, or O(1) if in-place sort is used
 *                   and input/output overhead is ignored.
 */

#include <bits/stdc++.h>
using namespace std;

int main() {
  /**
   * Functional Utility: Initializes variables and processes input for the problem.
   * This includes reading the total number of elements and the elements themselves.
   */
  int sum = 0; // Stores the total sum of all elements.
  int sum1 = 0; // Stores the sum of selected elements (the "twin" sum).
  int n; // Stores the number of elements in the array.
  cin >> n;
  int a[100]; // Array to store the input elements. Assuming N <= 100 as per common competitive programming constraints.

  /**
   * Block Logic: Reads N integers from input and calculates their total sum.
   * Pre-condition: 'n' has been read, and 'sum' is initialized to 0.
   * Invariant: At the start of each iteration, 'sum' holds the sum of 'a[0]' to 'a[i-1]'.
   *            'i' iterates from 0 to N-1.
   */
  for (int i = 0; i < n; i++) {
    cin >> a[i];
    sum = sum + a[i];
  }

  // Algorithm Core: Sorts the array in ascending order to facilitate the greedy selection.
  sort(a, a + n);

  int cnt = 0; // Counts the number of selected elements.

  /**
   * Block Logic: Greedily selects elements from the largest to the smallest
   *              until the sum of selected elements (`sum1`) is strictly greater
   *              than the sum of remaining elements (`sum`).
   * Pre-condition: 'a' is sorted in ascending order, 'sum' holds total sum,
   *                'sum1' is 0, and 'cnt' is 0.
   * Invariant: At the start of each iteration, 'sum1' is the sum of 'cnt' largest
   *            elements, and 'sum' is the sum of the remaining elements.
   *            'i' iterates from 'n-1' down to 0, representing the index of the
   *            current largest element being considered.
   */
  for (int i = n - 1; i >= 0; i--) {
    // Condition Check: Determines if the current set of selected elements
    // is sufficient (i.e., sum1 is already greater than sum).
    if (sum1 <= sum) {
      cnt++; // Increment count as another element is added to the "twin" group.
    } else {
      break; // The sum of selected elements is now strictly greater; no more elements needed.
    }
    sum1 = sum1 + a[i]; // Add the current largest element to the selected sum.
    sum = sum - a[i];   // Remove the current largest element from the total sum (effectively, sum of remaining elements).
  }

  // Output: Prints the minimum number of elements required.
  cout << cnt;
}
