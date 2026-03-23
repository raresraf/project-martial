/**
 * @file 50A.CPP
 * @brief This competitive programming solution attempts to solve the "Domino Piling" problem,
 *        which asks for the maximum number of 2x1 dominoes that can be placed on an M x N
 *        chessboard.
 *        Note: The standard optimal solution for this problem is simply `(M * N) / 2`
 *        due to each 2x1 domino covering exactly two squares. This implementation, however,
 *        attempts a simulation-based approach, which is more complex and potentially flawed
 *        for ensuring optimal 2x1 tiling.
 *
 * Algorithm: Imperfect Tiling Simulation.
 *            1. Initializes a 2D boolean array representing the chessboard to all `false` (empty).
 *            2. Iterates through the board, first attempting to place blocks that occupy 2x2 areas.
 *               If a 2x2 area's top-left cell is empty, it marks all four cells as occupied and
 *               increments a counter (this part appears to count 2x2 blocks rather than 2x1 dominoes optimally).
 *            3. Subsequently, it attempts to place 1x2 or 2x1 blocks (dominoes) in the remaining
 *               untiled areas along the edges.
 *            4. Prints the final count of placed blocks.
 *
 * Time Complexity: O(M * N), where M and N are the dimensions of the chessboard.
 *                  This is because the algorithm iterates over a significant portion,
 *                  or potentially all, of the M x N grid multiple times.
 * Space Complexity: O(M * N) for the `papan` (board) 2D boolean array that stores the state of each cell.
 */
#include <algorithm> // Included but not explicitly used in main logic.
#include <cmath>     // Included but not explicitly used in main logic.
#include <cstdio>    // Required for scanf and printf.
#include <cstring>   // Required for memset.
#include <iostream>  // Included but mainly using C-style I/O.
#include <map>       // Included but not explicitly used.
#include <queue>     // Included but not explicitly used.
#include <stack>     // Included but not explicitly used.
#include <string>    // Included but not explicitly used.
#include <utility>   // Included but not explicitly used.
#include <vector>    // Included but not explicitly used.

// Commonly used macros for competitive programming, though many are unused in this specific code.
#define fi first      ///< Macro for pair's first element.
#define se second     ///< Macro for pair's second element.
#define pb push_back  ///< Macro for vector's push_back method.
#define mp make_pair  ///< Macro for std::make_pair.
#define pi 2 * acos(0.0) ///< Macro for pi.
#define eps 1e-9      ///< Macro for a small epsilon value.
#define PII pair<int, int> ///< Macro for pair of integers.
#define PDD pair<double, double> ///< Macro for pair of doubles.
#define LL long long  ///< Macro for long long integer.

using namespace std;

int bawah, samping;    // 'bawah': height (M); 'samping': width (N) of the chessboard.
int x, y, ans;         // Loop counters and 'ans' to store the count of placed blocks.
bool papan[20][20];    // 2D boolean array representing the chessboard (max 20x20).
                       // `true` means occupied, `false` means empty.

/**
 * @brief Main function of the program. Reads board dimensions, attempts to tile it, and prints the result.
 * Functional Utility: Serves as the entry point, orchestrating the simulation of domino piling.
 */
int main() {
  // Reads the dimensions of the chessboard (height and width).
  scanf("%d %d", &bawah, &samping);
  
  // Functional Utility: Initializes all cells of the `papan` (board) to `false` (empty).
  memset(papan, false, sizeof(papan));
  ans = 0; // Initializes the counter for placed blocks to zero.

  // Block Logic: Attempts to tile the board by placing 2x2 blocks.
  // This approach is not directly suitable for optimal 2x1 domino tiling.
  // Invariant: For every non-occupied 2x2 top-left cell, 2 is added to `ans`,
  //            and a 2x2 area is marked as occupied.
  for (x = 1; x + 1 <= samping; x++) // Iterates through columns (samping).
    for (y = 1; y + 1 <= bawah; y++)   // Iterates through rows (bawah).
      if (!papan[x][y]) { // Checks if the top-left cell of a 2x2 block is empty.
        ans += 2;         // Increments answer by 2 (problematic for 2x1 dominoes).
        // Marks the 2x2 area as occupied.
        papan[x][y] = papan[x + 1][y] = papan[x][y + 1] = papan[x + 1][y + 1] =
            true;
      }

  // Block Logic: Attempts to tile remaining untiled cells along one dimension with 1x2 blocks.
  // This logic is also not directly optimal for 2x1 dominoes if previous tiling was incorrect.
  // Invariant: For every non-occupied cell along a column, 1 is added to `ans`,
  //            and a 1x2 area is marked as occupied.
  for (x = 1; x + 1 <= samping; x++) // Iterates through columns.
    if (!papan[x][bawah]) { // Checks if a cell at the edge of the row is empty.
      ans++;                // Increments answer by 1.
      papan[x][bawah] = papan[x + 1][bawah] = true; // Marks a 1x2 area as occupied.
    }

  // Block Logic: Attempts to tile remaining untiled cells along the other dimension with 2x1 blocks.
  // Similar logic as above, likely to cover remaining single cells or pairs.
  // Invariant: For every non-occupied cell along a row, 1 is added to `ans`,
  //            and a 2x1 area is marked as occupied.
  for (y = 1; y + 1 <= bawah; y++) // Iterates through rows.
    if (!papan[samping][y]) { // Checks if a cell at the edge of the column is empty.
      ans++;                  // Increments answer by 1.
      papan[samping][y] = papan[samping][y + 1] = true; // Marks a 2x1 area as occupied.
    }

  printf("%d
", ans); // Prints the final calculated count of placed blocks.
  return 0;            // Indicates successful program termination.
}
