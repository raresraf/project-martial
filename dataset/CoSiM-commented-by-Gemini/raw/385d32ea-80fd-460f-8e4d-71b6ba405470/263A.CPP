/**
 * @file 263A.CPP
 * @brief Solves the "Beautiful Matrix" problem (Codeforces 263A).
 *
 * @details The problem asks for the minimum number of moves (swapping adjacent
 * rows or columns) to move a single '1' in a 5x5 grid of zeroes to the center.
 * This is equivalent to calculating the Manhattan distance from the '1's
 * current position to the center of the grid.
 *
 * @algorithm The program iterates through a 5x5 grid to find the initial
 * coordinates (x, y) of the '1'. The center of the grid is at (3, 3) in the
 * 1-based indexing used here. The minimum number of moves is the Manhattan
 * distance, which is calculated as `abs(3 - x) + abs(3 - y)`.
 *
 * @note The code uses 1-based indexing for loops (1 to 5) on a 0-indexed
 * array `arr[5][5]` (indices 0 to 4), causing out-of-bounds access. While this
 * may appear to work on some systems, it is technically undefined behavior.
 *
 * @complexity
 * Time Complexity: O(1), since the input size is a constant 5x5 grid.
 * Space Complexity: O(1), for storing the 5x5 grid.
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

// Global array to store the 5x5 matrix.
int arr[5][5];

int main(int argc, char const *argv[]) {
  int x, y;
  /**
   * @brief Block Logic: Reads the 5x5 matrix and finds the coordinates of the '1'.
   * The loops use 1-based indexing from 1 to 5.
   */
  for (int i = 1; i <= 5; i++) {
    for (int j = 1; j <= 5; j++) {
      cin >> arr[i][j];
      if (arr[i][j] == 1) {
        x = i;
        y = j;
      }
    }
  }

  // Calculate the Manhattan distance to the center (3, 3).
  int test = abs(3 - x) + abs(3 - y);
  cout << test;
  return 0;
}