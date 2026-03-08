/**
 * @49c96202-5792-4b8c-a1ec-6478d5bc91a1/160A.CPP
 * @brief This program finds the minimum number of largest elements (`k`)
 *        whose sum is strictly greater than the sum of the remaining elements
 *        in an array.
 * Algorithm: Sorts the array (using an inefficient bubble sort-like approach)
 *            and then iteratively checks partitions from the largest elements.
 * Time Complexity: O(N^2) due to the bubble sort, and another O(N^2) for the nested loops
 *                  in the partitioning check, resulting in an overall O(N^2).
 * Space Complexity: O(N) for storing the array.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  // Declare variables:
  // n: The number of elements in the array.
  // temp: Temporary variable used for swapping during sorting.
  int n, temp;
  // Functional Utility: Read the number of elements.
  cin >> n;
  // Declare a dynamic array 'arr' of size 'n' to store integers.
  int arr[n];
  // Block Logic: Read 'n' integers into the array.
  for (int i = 0; i < n; i++)
    cin >> arr[i];

  // Block Logic: Sort the array in ascending order using a bubble sort-like algorithm.
  // Pre-condition: After this block, 'arr' will contain elements in non-decreasing order.
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      if (arr[i] > arr[j]) {
        temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
      }
    }
  }

  // Declare variables for partitioning logic:
  // total: Sum of the first (n-k) elements.
  // k: The number of largest elements being considered (starts from 1).
  // total1: Sum of the last 'k' largest elements.
  int total = 0, k = 1, total1 = 0;
  // Block Logic: Iterate to find the smallest 'k' such that the sum of the 'k' largest
  // elements is strictly greater than the sum of the remaining (n-k) elements.
  // Invariant: 'k' increases in each iteration, shifting the partition point.
  for (int j = 0; j < n; j++) {
    total = 0;  // Reset sum of smaller elements.
    total1 = 0; // Reset sum of larger elements.

    // Block Logic: Calculate the sum of the first (n-k) elements.
    for (int i = 0; i < n - k; i++)
      total += arr[i];
    // Block Logic: Calculate the sum of the last 'k' elements.
    // This loop effectively sums the last 'k' elements, but the condition `i > (n - k - 1)`
    // and `n - k` for `i` is a bit unusual; it seems intended to sum `arr[n-k]` to `arr[n-1]`.
    // A simpler way would be `for (int i = n - k; i < n; i++) total1 += arr[i];`
    for (int i = n - k; i > (n - k - 1); i--)
      total1 += arr[i];
    
    // Pre-condition: Check if the sum of the 'k' largest elements is strictly greater than the sum of the rest.
    if (total < total1) {
      cout << k << endl; // Functional Utility: Output the found 'k'.
      break;             // Exit loop once the condition is met.
    }
    k++; // Increment 'k' to consider one more largest element.
  }

  return 0; // Indicate successful program termination.
}
