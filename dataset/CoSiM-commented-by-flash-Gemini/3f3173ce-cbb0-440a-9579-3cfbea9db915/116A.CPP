/**
 * @file 116A.CPP
 * @brief This program processes a sequence of 2*N numbers. It reads N, then
 *        2*N integers into an array. It then iterates through this array,
 *        performing alternating addition and subtraction on adjacent elements,
 *        storing the result in the second element of each conceptual pair.
 *        Throughout this process, it tracks and ultimately prints the maximum
 *        value encountered in the modified array elements.
 *        Note: The code uses a C99-style Variable Length Array (VLA) `int num[n * 2];`
 *        which is not standard C++. It also modifies `num[i+1]` in a loop that increments `i`,
 *        which might lead to unexpected results or out-of-bounds access when `i` is `2*n - 1`.
 * Algorithm: Array Processing with Alternating Operations and Max Finding
 * Time Complexity: O(N), where N is the logical number of pairs (or `2*n` elements).
 *                  Each element is processed a constant number of times.
 * Space Complexity: O(N) to store the `2*n` numbers in an array.
 */

#include <bits/stdc++.h> // Includes various standard libraries, including iostream and climits (for INT_MIN).
using namespace std;

/**
 * @brief Main function to read numbers, perform alternating operations, and find the maximum resulting value.
 * @return 0 if the program executes successfully.
 */
int main() {
  int n; // Declare an integer to store the logical number of pairs.
  // Read the value of 'n' from standard input.
  cin >> n;
  
  // Initialize 'm' with the smallest possible integer value to track the maximum.
  int m = INT_MIN;
  
  // Declare a Variable Length Array (VLA) to store 2*n numbers.
  // This is a C99 feature and a non-standard C++ extension.
  int num[n * 2];
  
  /**
   * Block Logic: Reads 2*N integers into the 'num' array.
   * Invariant: After each iteration, one number is read and stored in 'num'.
   */
  for (int i = 0; i < n * 2; i++) {
    cin >> num[i]; // Read each number.
  }
  
  /**
   * Block Logic: Iterates through the 'num' array, performing alternating
   *              addition and subtraction on conceptual pairs, and updates the maximum 'm'.
   * Invariant: 'm' stores the maximum value encountered among the modified num[i+1] elements.
   *            'i' increments by 1, processing elements in an overlapping manner.
   *            When `i` is odd, `num[i+1]` is `num[i] - num[i+1]`. `num[i]` itself would have been the result of a previous `num[prev_i] + num[prev_i+1]` operation.
   *            Potential Issue: The loop condition `i < n * 2` means `i` can go up to `2*n - 1`.
   *            When `i = 2*n - 1`, `i+1 = 2*n`, which is out of bounds for `num[n*2]`.
   */
  for (int i = 0; i < n * 2; i++) {
    // If 'i' is an even index, perform addition with the next element.
    if (i % 2 == 0) {
      // Store the sum of num[i] and num[i+1] into num[i+1].
      num[i + 1] = num[i] + num[i + 1];
    } else {
      // If 'i' is an odd index, perform subtraction with the next element.
      // Store the difference of num[i] and num[i+1] into num[i+1].
      num[i + 1] = num[i] - num[i + 1];
    }
    // Update the maximum value 'm' if the modified num[i+1] is greater.
    if (num[i + 1] > m)
      m = num[i + 1];
  }
  // Print the final maximum value found.
  cout << m << endl;

  return 0; // Indicate successful program termination.
}
