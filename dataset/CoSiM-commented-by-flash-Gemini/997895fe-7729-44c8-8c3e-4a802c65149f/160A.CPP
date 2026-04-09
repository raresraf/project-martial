/**
 * @file 160A.CPP
 * @brief This program finds the minimum number of elements from a given set
 * whose sum is strictly greater than the sum of the remaining elements.
 * This is a classic greedy approach problem often encountered in competitive programming.
 *
 * Algorithm: Greedy approach with sorting.
 * 1. Read `n` integers and calculate their total sum.
 * 2. Sort the integers in non-decreasing order.
 * 3. Iterate from the largest elements (end of the sorted array) and accumulate their sum.
 * 4. Count the number of elements taken until their accumulated sum exceeds half of the total sum.
 * Time Complexity: O(N log N) primarily due to the sorting step, where N is the number of elements.
 * Space Complexity: O(N) to store the array of integers.
 */

#include <algorithm>
#include <iostream>

using namespace std;

/**
 * @brief Main function of the program.
 * Reads `n` numbers, finds the minimum count of largest numbers whose sum exceeds half the total sum.
 * @pre Input consists of an integer `n` (number of elements) followed by `n` integers.
 * @post Prints the minimum count of elements satisfying the condition.
 * @return 0 upon successful execution.
 */
int main() {
  // Functional Utility: 'n' for the number of elements, 'arr' to store elements, 'res' for total sum/current accumulated sum.
  int n, arr[101], res = 0;
  // Functional Utility: Reads the number of elements.
  cin >> n;
  // Block Logic: Reads `n` integers into the array `arr` and calculates their total sum.
  // Invariant: After each iteration, `arr[i]` is populated and `res` holds the sum of elements from `arr[0]` to `arr[i]`.
  for (int i = 0; i < n; i++) {
    cin >> arr[i];
    res += arr[i];
  }
  // Functional Utility: Sorts the array `arr` in non-decreasing order. This is crucial for the greedy strategy.
  sort(arr, arr + n);
  // Block Logic: (Original commented-out loop) This loop would print the sorted array.
  // for (int i = 0; i < n; i++) {
  //   cout << arr[i] << endl;
  // }

  // Functional Utility: Calculates the target sum, which is strictly greater than half of the total sum.
  int div = res / 2;
  // Functional Utility: Counter for the number of elements selected.
  int counter = 0;
  // Functional Utility: Resets `res` to 0 to accumulate the sum of largest elements.
  res = 0;
  // Block Logic: Iterates through the sorted array from the largest elements downwards.
  // Invariant: `res` holds the sum of the largest `counter` elements processed so far.
  for (int i = n - 1; i >= 0; i--) {
    res += arr[i]; // Functional Utility: Adds the current largest element to the accumulated sum.
    counter++;     // Functional Utility: Increments the count of selected elements.
    // Block Logic: Checks if the accumulated sum of largest elements has exceeded half of the total sum.
    // Pre-condition: `res` holds the sum of `counter` largest elements.
    if (res > div)
      break; // Functional Utility: Breaks the loop as the condition is met.
  }
  // Functional Utility: Prints the minimum number of elements whose sum exceeds half the total sum.
  cout << counter << endl;

  return 0;
}
