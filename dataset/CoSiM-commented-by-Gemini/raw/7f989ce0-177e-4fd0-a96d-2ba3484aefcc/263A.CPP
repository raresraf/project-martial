/**
 * @file 263A.CPP
 * @brief Calculates the number of moves to center a '1' in a 5x5 matrix.
 *
 * This program reads a 5x5 matrix that contains exactly one '1' and all other
 * elements as '0's. It calculates the minimum number of single-step moves
 * (up, down, left, or right) required to move the '1' to the center of the
 * matrix, which is at position (3,3) in a 1-indexed grid. This is equivalent
 * to calculating the Manhattan distance.
 *
 * @algorithm
 * 1. Find the coordinates (row, col) of the '1' in the grid.
 * 2. Calculate the Manhattan distance between these coordinates and the center (3,3).
 *    Distance = |3 - row| + |3 - col|.
 *
 * @complexity
 * Time: O(1), as the matrix size is fixed at 5x5. In the worst case, it reads all 25 elements.
 * Space: O(1), as it uses a fixed amount of space for variables. The vector 'ar'
 *        is not fully populated due to early loop termination.
 */
#include <iostream>
#include <vector>
#include <cmath> // For std::abs

using namespace std;

// --- Competitive Programming Boilerplate ---
#define fo(i, n) for (i = 0; i < n; i++)
#define deb(x) cout << #x << "=" << x << endl
typedef vector<int> vi;
typedef vector<vi> vvi;

/**
 * @brief Main function to find the '1' and calculate its distance to the center.
 */
int main() {
  int i, j;
  vvi ar; // The 2D vector to store the matrix, though it's not fully used.
  int xx, y; // To store the 0-indexed row (xx) and column (y) of the '1'.
  int f = 1; // A flag to signal when the '1' has been found to break the outer loop.

  // Pre-condition: Standard input contains a 5x5 matrix of 0s and one 1.
  // Block Logic: Loop through the 5x5 grid to find the location of the '1'.
  fo(i, 5) {
    vi x(5);
    fo(j, 5) {
      cin >> x[j];
      // When the '1' is found, store its 0-indexed coordinates.
      if (x[j] == 1) {
        xx = i;
        y = j;
        f = 0; // Set flag to indicate '1' is found.
        break; // Exit the inner loop.
      }
    }
    // If the '1' was found in the inner loop, break from the outer loop too.
    if (!f)
      break;
    // This push_back is often skipped due to the break.
    ar.push_back(x);
  }

  // Invariant: 'xx' and 'y' hold the 0-indexed coordinates of the '1'.

  // Convert coordinates to 1-indexed to match the problem's center of (3,3).
  xx++;
  y++;

  // Calculate the Manhattan distance: |target_row - current_row| + |target_col - current_col|.
  // Here, the target is (3,3).
  int row_moves = abs(3 - xx);
  int col_moves = abs(3 - y);
  
  // Print the total number of moves.
  cout << row_moves + col_moves;
  
  return 0;
}
