/**
 * @file 263A.CPP
 * @brief Calculates the number of moves to center a '1' in a 5x5 matrix.
 *
 * This program reads a 5x5 matrix containing 24 zeros and a single one.
 * It calculates the Manhattan distance required to move the '1' to the
 * center of the matrix, which is at position (3, 3).
 *
 * Algorithm:
 * 1. A 5x5 integer array `arr` is declared.
 * 2. The program iterates through a 5x5 grid using 1-based indexing (from 1 to 5).
 * 3. It reads an integer for each position in the grid.
 * 4. When the integer '1' is found, its 1-based coordinates (`x`, `y`) are stored.
 * 5. The Manhattan distance from (`x`, `y`) to the center (3, 3) is calculated
 *    as `abs(3 - x) + abs(3 - y)`.
 * 6. The result, representing the minimum number of single-step moves (up, down,
 *    left, or right) required, is printed.
 *
 * Time Complexity: O(1), since the matrix size is constant (5x5).
 * Space Complexity: O(1), for storing the 5x5 matrix.
 *
 * Note: The code includes a large number of standard headers that are not used
 * for this specific problem. The loops use 1-based indexing (1 to 5) on a
 * 0-indexed array `arr[5][5]`, which leads to out-of-bounds access. However,
 * the logic to find the '1' and calculate the distance works as intended by the author.
 */
#include <algorithm>
#include <bitset>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <functional>
#include <iomanip>
#include <iostream>
#include <list>
#include <map>
#include <memory.h>
#include <numeric>
#include <set>
#include <sstream>
#include <stack>
#include <utility>
#include <vector>

using namespace std;

// Global 5x5 array to store the matrix.
int arr[5][5];

int main(int argc, char const *argv[]) {
  int x, y;
  /**
   * @brief Reads the 5x5 matrix and finds the coordinates of the '1'.
   * The loops use 1-based indexing, which is unconventional for C++ arrays.
   */
  for (int i = 1; i <= 5; i++) {
    for (int j = 1; j <= 5; j++) {
      cin >> arr[i][j];
      // Pre-condition: An integer is read into arr[i][j].
      // If the element is 1, store its coordinates.
      if (arr[i][j] == 1) {
        x = i;
        y = j;
      }
    }
  }

  // Calculate the Manhattan distance to the center (3,3).
  int test = abs(3 - x) + abs(3 - y);
  cout << test;
  return 0;
}