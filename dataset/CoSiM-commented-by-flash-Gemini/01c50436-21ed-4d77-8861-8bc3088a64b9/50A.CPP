/**
 * @file 50A.CPP
 * @brief Solution to the "Domino piling" problem.
 * This program calculates the maximum number of 1x2 or 2x1 dominoes that can be placed on an M x N rectangular board.
 * Algorithm: Direct calculation of floor(M * N / 2).
 * Time Complexity: O(1)
 * Space Complexity: O(1)
 */

#include <bits/stdc++.h>

using namespace std;

int main() {
  /**
   * Functional Utility: Optimizes I/O operations for competitive programming.
   * Disables synchronization with C's stdio and unties cin/cout.
   */
  ios::sync_with_stdio(false);
  cin.tie(nullptr);
  cout.tie(nullptr);
  
  int n, m;
  // Read the dimensions of the board (M and N).
  scanf("%d %d", &n, &m);
  
  // Block Logic: Calculate the number of dominoes that can fit into the main rectangular part of the board.
  // Invariant: 'i' stores the cumulative count of dominoes.
  int i = m / 2; // Calculate how many full 2-unit rows (or columns) of dominoes fit along one dimension.
  i *= n;        // Multiply by the other dimension to get the area covered by these full rows/columns, divided by 2.
  
  // Block Logic: Check for a remaining 1-unit wide strip and if it can accommodate more dominoes.
  // Precondition: 'x' is the remainder of m divided by 2 (0 or 1), 'y' is n divided by 2.
  int x = m % 2, y = n / 2;
  // If there is a remaining 1-unit strip along 'm' and at least one full 2-unit row/column along 'n',
  // then additional dominoes can be placed in this remaining area.
  if (x > 0 && y > 0)
    i += x * y; // Add the number of dominoes that fit into this remaining 1-unit strip (x * (n/2)).
  
  // Print the total maximum number of dominoes.
  printf("%d\n", i);
  return 0;
}
