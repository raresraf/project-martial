/**
 * @file 50A.CPP
 * @brief Solution for Codeforces Problem 50A - "Domino Piling".
 *
 * @details
 * This program aims to find the maximum number of 2x1 dominoes that can be
 * placed on a rectangular MxN board. The standard mathematical solution is
 * to calculate floor((M * N) / 2).
 *
 * This specific implementation, however, does not use the direct formula.
 * Instead, it employs a complex procedural tiling algorithm.
 *
 * Algorithm (as implemented):
 * 1. A boolean grid `papan` is used to track covered cells.
 * 2. A first pass attempts to tile the grid with 2x2 blocks. For each
 *    successful block placement, it assumes two dominoes are placed.
 * 3. Subsequent passes attempt to tile the remaining uncovered cells along the
 *    edges of the board with single dominoes.
 *
 * Time Complexity: O(M * N), where M and N are the board dimensions, due to the loops iterating over the grid.
 * Space Complexity: O(1), as the boolean grid `papan` has a fixed size (20x20), not
 *                   dependent on the input dimensions.
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <queue>
#include <stack>
#include <string>
#include <utility>
#include <vector>

// Collection of common competitive programming macros.
#define fi first
#define se second
#define pb push_back
#define mp make_pair
#define pi 2 * acos(0.0)
#define eps 1e-9
#define PII pair<int, int>
#define PDD pair<double, double>
#define LL long long

using namespace std;

// `bawah` corresponds to height (M), `samping` to width (N).
int bawah, samping, x, y, ans;
// `papan` is a boolean board to mark tiled areas.
bool papan[20][20];

int main() {
  scanf("%d %d", &bawah, &samping);
  // Initialize the board as completely untiled.
  memset(papan, false, sizeof(papan));
  ans = 0;

  /**
   * Block Logic: Main tiling phase.
   * This loop attempts to tile the board's interior with 2x2 blocks.
   * For each 2x2 area it can place, it adds 2 to the domino count,
   * implying two dominoes covering a 2x2 square.
   */
  for (x = 1; x + 1 <= samping; x++)
    for (y = 1; y + 1 <= bawah; y++)
      if (!papan[x][y]) {
        ans += 2;
        papan[x][y] = papan[x + 1][y] = papan[x][y + 1] = papan[x + 1][y + 1] =
            true;
      }

  /**
   * Block Logic: Edge tiling (vertical).
   * This loop attempts to tile remaining spaces along the bottom edge
   * with vertically oriented 2x1 dominoes.
   */
  for (x = 1; x + 1 <= samping; x++)
    if (!papan[x][bawah]) {
      ans++;
      papan[x][bawah] = papan[x + 1][bawah] = true;
    }

  /**
   * Block Logic: Edge tiling (horizontal).
   * This loop attempts to tile remaining spaces along the right edge
   * with horizontally oriented 1x2 dominoes.
   */
  for (y = 1; y + 1 <= bawah; y++)
    if (!papan[samping][y]) {
      ans++;
      papan[samping][y] = papan[samping][y + 1] = true;
    }

  printf("%d
", ans);
  return 0;
}
