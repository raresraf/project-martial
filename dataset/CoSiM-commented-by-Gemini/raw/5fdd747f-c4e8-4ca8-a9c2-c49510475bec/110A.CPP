/**
 * @file 110A.CPP
 * @brief Counts the number of pairs (i, j) where the sum of row `i` is greater
 *        than the sum of column `j` in a square matrix.
 * @details This program reads an N x N matrix, calculates the sum of each row and
 * each column, and then counts how many times a row's sum is strictly greater
 * than a column's sum.
 *
 * @note The logic in this file does not solve the Codeforces problem 110A
 * ("Nearly Lucky Number"). It solves a different, matrix-related problem.
 *
 * @algorithm Matrix Summation and Pairwise Comparison
 * 1. Read the dimension `n` of the square matrix.
 * 2. Initialize two vectors, `R` and `C`, of size `n` to store row and column sums.
 * 3. Use nested loops to read each element of the matrix. For each element `a[i][j]`,
 *    add its value to `R[i]` and `C[j]`.
 * 4. After all sums are computed, use another pair of nested loops to iterate through
 *    all `n*n` possible pairs of row sums `R[i]` and column sums `C[j]`.
 * 5. For each pair, if `R[i] > C[j]`, increment a counter.
 * 6. Print the final count.
 *
 * @complexity
 * - Time: O(N^2), where N is the dimension of the matrix. The program has two
 *   separate sections of O(N^2) nested loops (one for summation, one for comparison).
 * - Space: O(N), to store the row and column sum vectors.
 */
#include <algorithm>
#include <iostream>
#include <map>
#include <queue>
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>
using namespace std;

int main() {
  // `freopen` is used for local testing to read from a file instead of stdin.
  // freopen("A.txt","r",stdin);

  // n: dimension of the square matrix.
  // i, j: loop counters.
  // cnt: counter for pairs where row_sum > col_sum.
  // t: temporary variable to store each matrix element.
  long int n, i, j, cnt = 0, t;
  cin >> n;

  // R: vector to store the sum of each row. Initialized to zeros.
  // C: vector to store the sum of each column. Initialized to zeros.
  vector<int> R(n, 0), C(n, 0);

  /**
   * @brief Read the n x n matrix and compute row and column sums simultaneously.
   * The `cin >> t` inside the loop condition is a compact way to control the loop
   * while also reading the value.
   */
  for (i = 0; i < n; i++)
    for (j = 0; j < n && cin >> t; j++) {
      R[i] += t; // Add the element to its row's sum.
      C[j] += t; // Add the element to its column's sum.
    }

  /**
   * @brief Compare every row sum with every column sum.
   * This block iterates through all N*N possible (row, column) pairings.
   */
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      // Pre-condition: Check if the sum of row `i` is greater than the sum of column `j`.
      if (R[i] > C[j])
        cnt++;

  // Print the total count of such pairs.
  cout << cnt << "\n";
  return 0;
}
