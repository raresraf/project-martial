/**
 * @file 263A.CPP
 * @brief Solves a matrix problem to find a single '1' in a 5x5 matrix and calculates its Manhattan distance to the center (3,3).
 *
 * Algorithm: Iterative search for the target element followed by Manhattan distance calculation.
 * Time Complexity: O(R*C) where R and C are the dimensions of the matrix. For a fixed 5x5 matrix, this is O(1).
 * Space Complexity: O(R*C) for storing the matrix. For a fixed 5x5 matrix, this is O(1).
 */

#include <bits/stdc++.h>
using namespace std;

#define ll long long int
#define mp make_pair
#define pb push_back
#define fi first
#define si second
/**
 * @brief Optimizes C++ standard input/output streams.
 *
 * Functional Utility: Decouples C++ streams from standard C streams and unties cin from cout,
 * improving I/O performance which is crucial in competitive programming.
 */
#define fastIO                                                                 \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(NULL);
/**
 * @brief Macro for reading 'n' elements into a 1D array 'a' (0-indexed).
 * @param a The array to read into.
 * @param n The number of elements to read.
 */
#define F(a, n)                                                                \
  for (int i = 0; i < n; i++) {                                                \
    cin >> a[i];                                                               \
  }
/**
 * @brief Macro for reading 'n' elements into a 1D array 'a' (1-indexed).
 * @param a The array to read into.
 * @param n The number of elements to read.
 */
#define F1(a, n)                                                               \
  for (int i = 1; i <= n; i++) {                                               \
    cin >> a[i];                                                               \
  }
/**
 * @brief Macro for printing 'n' elements of a 1D array 'a' (0-indexed).
 * @param a The array to print.
 * @param n The number of elements to print.
 */
#define P(a, n)                                                                \
  for (int i = 0; i < n; i++) {                                                \
    cout << a[i] << ' ';                                                       \
  }                                                                            \
  cout << endl;
/**
 * @brief Macro for printing 'n' elements of a 1D array 'a' (1-indexed).
 * @param a The array to print.
 * @param n The number of elements to print.
 */
#define P1(a, n)                                                               \
  for (int i = 1; i <= n; i++) {                                               \
    cout << a[i] << ' ';                                                       \
  }                                                                            \
  cout << endl;
/**
 * @brief Macro for reading 'n x m' elements into a 2D array 'a' (0-indexed).
 * @param a The 2D array to read into.
 * @param n The number of rows.
 * @param m The number of columns.
 */
#define NF(a, n, m)                                                            \
  for (int i = 0; i < n; i++) {                                                \
    for (int j = 0; j < m; j++) {                                              \
      cin >> a[i][j];                                                          \
    }                                                                          \
  }
/**
 * @brief Macro for reading 'n x m' elements into a 2D array 'a' (1-indexed).
 * @param a The 2D array to read into.
 * @param n The number of rows.
 * @param m The number of columns.
 */
#define NF1(a, n, m)                                                           \
  for (int i = 1; i <= n; i++) {                                               \
    for (int j = 1; j <= m; j++) {                                             \
      cin >> a[i][j];                                                          \
    }                                                                          \
  }
/**
 * @brief Macro for printing 'n x m' elements of a 2D array 'a' (0-indexed).
 * @param a The 2D array to print.
 * @param n The number of rows.
 * @param m The number of columns.
 */
#define PNF(a, n, m)                                                           \
  for (int i = 0; i < n; i++) {                                                \
    for (int j = 0; j < m; j++) {                                              \
      cout << a[i][j] << ' ';                                                  \
    }                                                                          \
    cout << endl;                                                              \
  }                                                                            \
  cout << endl;
/**
 * @brief Macro for printing 'n x m' elements of a 2D array 'a' (1-indexed).
 * @param a The 2D array to print.
 * @param n The number of rows.
 * @param m The number of columns.
 */
#define PNF1(a, n, m)                                                          \
  for (int i = 1; i <= n; i++) {                                               \
    for (int j = 1; j <= m; j++) {                                             \
      cout << a[i][j] << ' ';                                                  \
    }                                                                          \
    cout << endl;                                                              \
  }                                                                            \
  cout << endl;
#define AS 200001
#define mod 1000000007

/**
 * @brief Main function where the program execution begins.
 *
 * Block Logic: Initializes a 5x5 matrix, reads its elements, finds the position of '1',
 * and then calculates the Manhattan distance to the center (3,3).
 */
int main() {
  fastIO
  int n = 5;
  vector<vector<int>> matrix(n, vector<int>(n, 0));
  
  // Block Logic: Reads the 5x5 matrix from standard input.
  NF(matrix, 5, 5);
  cout << endl;
  // PNF(matrix, 5, 5); // Debugging line to print the matrix, currently commented out.
  
  int row, col;
  // Block Logic: Iterates through the matrix to find the element with value '1'.
  // Invariant: 'row' and 'col' will store the 1-indexed coordinates of the '1'.
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (matrix[i][j] == 1) {
        // Inline: Converts 0-indexed (i, j) to 1-indexed (row, col) coordinates.
        row = i + 1, col = j + 1;
      }
    }
  }
  // Output: Calculates and prints the Manhattan distance from the found '1' (row, col)
  // to the center of the 5x5 matrix, which is considered (3,3) in 1-indexed terms.
  cout << abs(3 - row) + abs(3 - col);
}