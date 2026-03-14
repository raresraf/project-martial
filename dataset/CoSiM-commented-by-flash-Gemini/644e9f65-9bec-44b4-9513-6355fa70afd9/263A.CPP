/*
        Author   : Abhinav

        Modified : 24-08-2018 01:41:12 AM

*/

/**
 * @644e9f65-9bec-44b4-9513-6355fa70afd9/263A.CPP
 * @brief This program solves problem 263A, "Beautiful Matrix". It reads a fixed 5x5 matrix
 * where exactly one cell contains the value 1. The goal is to determine the minimum
 * number of moves (Manhattan distance) required to bring the cell containing 1
 * to the center of the matrix (position (2,2) for 0-indexed coordinates).
 *
 * Algorithm: Matrix traversal to locate the '1', followed by Manhattan distance calculation.
 * Time Complexity: O(1) as the matrix size (5x5) is constant.
 * Space Complexity: O(1) for storing the fixed-size matrix and variables.
 */

#include <bits/stdc++.h> // Includes many standard C++ library headers.

using namespace std; // Uses the standard namespace.

typedef long long ll;     // @brief Defines 'll' as a shorthand for 'long long' for convenience.
typedef vector<ll> vi;   // @brief Defines 'vi' as a shorthand for a vector of 'long long' (unused in this solution).
typedef map<ll, ll> mllmp; // @brief Defines 'mllmp' as a shorthand for a map from 'long long' to 'long long' (unused in this solution).

/**
 * @brief Macro to speed up I/O operations in C++ by disabling synchronization
 * with C's stdio and untieing cin from cout.
 */
#define faster                                                                 \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(0);
#define make_pair mp // @brief Defines 'make_pair' as a shorthand for 'mp' (potentially from another library or custom, unused).

const ll mod = 1e9 + 7; // @brief A constant 'long long' value, likely for modulo operations (unused in this solution).

/**
 * @brief Macro for a for-loop that iterates from j up to (but not including) k.
 * @param i Loop variable.
 * @param j Starting value (inclusive).
 * @param k Ending value (exclusive).
 */
#define rep(i, j, k) for (ll i = j; i < k; i++)
/**
 * @brief Macro for a reverse for-loop that iterates from j down to (but not including) k.
 * @param i Loop variable.
 * @param j Starting value (inclusive).
 * @param k Ending value (exclusive).
 */
#define repv(i, j, k) for (ll i = j; i > k; i--)

int main() {
  faster; // Apply I/O optimization.
  ll x, y; // Declares 'x' and 'y' to store the 0-indexed row and column of the cell containing '1'.
  
  // Declares a 2D vector (dynamically sized 5x5 matrix) to store the integer values.
  vector<vector<int>> v(5, vector<int>(5)); 
  
  /**
   * Block Logic: Reads the 5x5 matrix and finds the coordinates of the cell containing the value 1.
   * Invariant: After this nested loop, 'x' and 'y' store the 0-indexed row and column
   * of the cell containing the value 1.
   */
  rep(i, 0, 5) rep(j, 0, 5) { // Nested loops iterating through rows (0-4) and columns (0-4).
    cin >> v[i][j]; // Reads the integer value for the current cell.
    if (v[i][j] == 1) { // Inline: If the current cell contains the value 1.
      x = i; // Stores the row index of '1'.
      y = j; // Stores the column index of '1'.
    }
  }
  
  /**
   * Block Logic: Calculates and prints the minimum number of moves (Manhattan distance)
   * required to bring the cell containing '1' to the center (2,2).
   * Invariant: The Manhattan distance is correctly computed based on the relative position to (2,2).
   */
  if (x == 2 && y == 2) // If '1' is already at the center (2,2).
    cout << "0";       // 0 moves needed.
  else if (x == 2)     // If '1' is in the center row, but not center column.
    cout << abs(y - 2); // Moves needed are absolute difference in columns.
  else if (y == 2)     // If '1' is in the center column, but not center row.
    cout << abs(x - 2); // Moves needed are absolute difference in rows.
  else                 // If '1' is neither in the center row nor center column.
    // Moves needed are the sum of absolute differences in rows and columns (Manhattan distance).
    cout << abs(y - 2) + abs(x - 2); 
  return 0; // Indicates successful program termination.
}