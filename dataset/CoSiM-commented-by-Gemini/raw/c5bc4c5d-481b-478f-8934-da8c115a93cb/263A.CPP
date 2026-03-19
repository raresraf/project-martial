/**
 * @file raw/c5bc4c5d-481b-478f-8934-da8c115a93cb/263A.CPP
 * @brief Solves a grid manipulation problem by calculating the Manhattan distance.
 * The program reads a 5x5 grid and determines the minimum number of moves
 * required to move the single digit '1' to the center of the grid (position 3,3).
 *
 * @algorithm
 * 1. Iterate through a 5x5 grid to locate the coordinates of the value '1'.
 * 2. Calculate the Manhattan distance from the found coordinates to the center (3,3).
 * The Manhattan distance |x1 - x2| + |y1 - y2| gives the minimum number of swaps
 * to move an element from (x1, y1) to (x2, y2) on a grid.
 *
 * @time_complexity O(1), as the grid size is fixed to 5x5.
 * @space_complexity O(1), as storage is for a few variables regardless of input size.
 */
#include <iostream>

using namespace std;

int main() {
  int idx, idy, temp;

  /**
   * @block
   * Pre-condition: The input stream contains 25 integers representing the 5x5 grid.
   * Invariant: After each outer loop iteration `i`, the first `i` rows have been scanned.
   * This nested loop iterates through the 5x5 grid to find the coordinates of the element '1'.
   */
  for (int i = 1; i < 6; i++) {
    for (int j = 1; j < 6; j++) {
      cin >> temp;
      if (temp == 1) {
        // Store the 1-based coordinates of the element '1'.
        idx = i;
        idy = j;
      }
    }
  }

  /**
   * @calculation
   * Calculates the Manhattan distance between the element's position (idx, idy)
   * and the center of the grid (3, 3). This represents the minimum number of
   * single-step (up, down, left, or right) moves required.
   */
  cout << (abs(3 - idx) + abs(3 - idy)) << endl;
}
