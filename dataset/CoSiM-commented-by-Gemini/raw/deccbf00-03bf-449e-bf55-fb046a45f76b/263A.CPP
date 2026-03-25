#include <algorithm>
#include <iostream>
#include <cmath> // For std::abs

/**
 * @file 263A.CPP
 * @brief Solution for a grid-based problem, "Beautiful Matrix".
 *
 * Problem Statement:
 * You are given a 5x5 matrix containing 24 zeroes and a single one. The goal
 * is to move the '1' to the center of the matrix (at position (3,3)). The
 * allowed moves are swapping two adjacent rows or swapping two adjacent
 * columns. Find the minimum number of moves required.
 *
 * Algorithm:
 * The minimum number of moves required is the Manhattan distance between the
 * current coordinates of the '1' and the target center coordinates.
 *
 * The Manhattan distance between two points (x1, y1) and (x2, y2) is
 * `|x1 - x2| + |y1 - y2|`.
 *
 * In this problem, the target is (3,3). If the '1' is at (i, j), the number
 * of row swaps needed is `|i - 3|` and the number of column swaps is `|j - 3|`.
 * The total minimum moves is their sum.
 *
 * 1. Iterate through the 5x5 matrix to find the coordinates (row, col) of the '1'.
 * 2. Calculate the Manhattan distance: `abs(row - 3) + abs(col - 3)`.
 * 3. Print the result.
 *
 * Time Complexity: O(1), as the matrix size is fixed at 5x5 (25 reads).
 * Space Complexity: O(1).
 */

using namespace std;

// Global variables to store input, coordinates of '1', and the result.
int b, sti, stj, pas;

int main() {
  // Block Logic: Iterate through a 5x5 grid using 1-based indexing
  // to find the location of the '1'.
  for (int i = 1; i <= 5; i++) {
    for (int j = 1; j <= 5; j++) {
      cin >> b;
      // Pre-condition: Check if the current element is the '1'.
      if (b == 1) {
        // Store the 1-indexed row and column of the '1'.
        sti = i;
        stj = j;
      }
    }
  }

  // Calculate the result using the Manhattan distance formula.
  // abs(sti - 3) gives the number of row swaps (vertical moves).
  // abs(stj - 3) gives the number of column swaps (horizontal moves).
  pas = abs(sti - 3) + abs(stj - 3);
  
  cout << pas;
}
