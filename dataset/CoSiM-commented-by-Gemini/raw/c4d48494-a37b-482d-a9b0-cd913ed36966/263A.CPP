/**
 * @file 263A.CPP
 * @brief Solution for a competitive programming problem.
 * @details This program reads a 5x5 matrix and calculates the Manhattan distance 
 *          from the cell containing the value '1' to the center of the matrix (3,3).
 *
 * @section Algorithm
 * The algorithm iterates through the 5x5 matrix to find the coordinates (row, col) of the element '1'.
 * Once found, it calculates the Manhattan distance to the center (3,3) using the formula:
 * |3 - row| + |3 - col|.
 *
 * @section Complexity
 * - Time Complexity: O(1), as the matrix size is constant (5x5). The nested loops run a fixed 25 times.
 * - Space Complexity: O(1), as the matrix size is constant (5x5).
 */
#include <bits/stdc++.h>
using namespace std;

// These macros are common in competitive programming for brevity, but are not best practice for production code.
#define ll long long int
#define mp make_pair
#define pb push_back
#define fi first
#define si second
/**
 * @brief Macro to disable synchronization with C standard streams and untie cin/cout.
 * @details This is a common optimization in competitive programming to speed up I/O operations.
 */
#define fastIO                                                                 
  ios_base::sync_with_stdio(false);                                            
  cin.tie(NULL);
// Macro for reading a 1D array
#define F(a, n)                                                                
  for (int i = 0; i < n; i++) {                                                
    cin >> a[i];                                                               
  }
// Macro for reading a 1D array (1-based indexing)
#define F1(a, n)                                                               
  for (int i = 1; i <= n; i++) {                                               
    cin >> a[i];                                                               
  }
// Macro for printing a 1D array
#define P(a, n)                                                                
  for (int i = 0; i < n; i++) {                                                
    cout << a[i] << ' ';                                                       
  }                                                                            
  cout << endl;
// Macro for printing a 1D array (1-based indexing)
#define P1(a, n)                                                               
  for (int i = 1; i <= n; i++) {                                               
    cout << a[i] << ' ';                                                       
  }                                                                            
  cout << endl;
// Macro for reading a 2D array
#define NF(a, n, m)                                                            
  for (int i = 0; i < n; i++) {                                                
    for (int j = 0; j < m; j++) {                                              
      cin >> a[i][j];                                                          
    }                                                                          
  }
// Macro for reading a 2D array (1-based indexing)
#define NF1(a, n, m)                                                           
  for (int i = 1; i <= n; i++) {                                               
    for (int j = 1; j <= m; j++) {                                             
      cin >> a[i][j];                                                          
    }                                                                          
  }
// Macro for printing a 2D array
#define PNF(a, n, m)                                                           
  for (int i = 0; i < n; i++) {                                                
    for (int j = 0; j < m; j++) {                                              
      cout << a[i][j] << ' ';                                                  
    }                                                                          
    cout << endl;                                                              
  }                                                                            
  cout << endl;
// Macro for printing a 2D array (1-based indexing)
#define PNF1(a, n, m)                                                          
  for (int i = 1; i <= n; i++) {                                               
    for (int j = 1; j <= m; j++) {                                             
      cout << a[i][j] << ' ';                                                  
    }                                                                          
    cout << endl;                                                              
  }                                                                            
  cout << endl;
#define AS 200001
#define mod 1000000007
/**
 * @brief Main function to execute the logic.
 */
int main() {
  // fastIO macro call for fast I/O
  fastIO;
  
  // The problem is defined on a 5x5 matrix.
  int n = 5;
  vector<vector<int>> matrix(n, vector<int>(n, 0));
  
  // Pre-condition: Reads a 5x5 integer matrix from standard input.
  NF(matrix, 5, 5);
  cout << endl;

  // The following line is commented out, likely for debugging purposes to print the matrix.
  // PNF(matrix, 5, 5);
  
  int row, col;
  /**
   * Block Logic: Iterates through the matrix to find the element '1'.
   * Invariant: The loops continue until the '1' is found and its 0-indexed coordinates (i, j) are recorded.
   */
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (matrix[i][j] == 1) {
        // Post-condition: Stores the 1-based coordinates of the '1'.
        row = i + 1, col = j + 1;
      }
    }
  }

  // Calculate and print the Manhattan distance to the center (3,3).
  cout << abs(3 - row) + abs(3 - col);
}
