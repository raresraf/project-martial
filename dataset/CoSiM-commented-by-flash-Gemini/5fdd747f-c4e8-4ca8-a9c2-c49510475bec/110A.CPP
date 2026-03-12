/**
 * @5fdd747f-c4e8-4ca8-a9c2-c49510475bec/110A.CPP
 * @brief This program analyzes an N x N matrix of integers. For each possible
 * combination of a row 'i' and a column 'j', it compares the sum of elements
 * in row 'i' with the sum of elements in column 'j'. The objective is to
 * count how many times the sum of a given row 'i' is strictly greater than
 * the sum of a given column 'j'.
 *
 * Algorithm: Two-pass approach. First pass calculates all row and column sums.
 * Second pass performs N*N comparisons of these sums.
 * Time Complexity: O(N^2) where N is the dimension of the square matrix.
 * Space Complexity: O(N) for storing row sums and column sums.
 */

#include <algorithm> // Included but not explicitly used in this snippet.
#include <iostream>  // For standard input/output operations (cin, cout).
#include <map>       // Included but not explicitly used in this snippet.
#include <queue>     // Included but not explicitly used in this snippet.
#include <stdio.h>   // Included but iostream is primarily used.
#include <string.h>  // Included but not explicitly used in this snippet.
#include <string>    // Included but not explicitly used in this snippet.
#include <vector>    // For using std::vector.
using namespace std; // Uses the standard namespace.

int main() {
  // freopen("A.txt","r",stdin); // Commented out: typical for redirecting input in competitive programming.
  long int n, i, j, cnt = 0, t; // Declares variables: n (matrix dimension), i, j (loop counters),
                                // cnt (counter for R[i] > C[j] condition), t (temporary for input).
  cin >> n; // Reads the dimension N of the square matrix from standard input.
  
  // Declares two vectors of size 'n', 'R' for row sums and 'C' for column sums,
  // both initialized with zeros.
  vector<int> R(n, 0), C(n, 0); 

  /**
   * Block Logic: First pass - reads the N x N matrix elements and simultaneously
   * calculates the sum for each row and each column.
   * Invariant: After this nested loop, R[i] contains the sum of row 'i' and C[j]
   * contains the sum of column 'j'.
   */
  for (i = 0; i < n; i++) { // Outer loop iterates through rows.
    for (j = 0; j < n && cin >> t; j++) { // Inner loop iterates through columns, reading each element 't'.
      R[i] += t; // Inline: Adds the current element 't' to the sum of row 'i'.
      C[j] += t; // Inline: Adds the current element 't' to the sum of column 'j'.
    }
  }

  /**
   * Block Logic: Second pass - compares every row sum with every column sum.
   * Invariant: 'cnt' accurately reflects the number of pairs (row_sum, col_sum)
   * where row_sum is strictly greater than col_sum.
   */
  for (i = 0; i < n; i++) { // Outer loop iterates through row sums.
    for (j = 0; j < n; j++) { // Inner loop iterates through column sums.
      // Inline: Checks if the current row sum is strictly greater than the current column sum.
      if (R[i] > C[j])
        cnt++; // Increments the counter if the condition is met.
    }
  }

  cout << cnt << "\n"; // Prints the final count of pairs where row sum > column sum.
  return 0;            // Indicates successful program termination.
}
