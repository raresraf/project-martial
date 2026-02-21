/**
 * @file 50A.CPP
 * @brief Calculates the maximum number of dominoes on a rectangular board.
 *
 * This program solves the problem of finding the maximum number of 2x1 dominoes
 * that can be placed on a rectangular board of size M x N without any overlap.
 *
 * Algorithm:
 * 1. Read the integer dimensions of the board, M and N.
 * 2. Calculate the total area of the board by multiplying M and N.
 * 3. The area of a single domino is 2 (2x1).
 * 4. The maximum number of dominoes that can fit is the total board area
 *    divided by the area of one domino. Integer division handles the math
 *    correctly, as any single remaining square cannot fit a domino.
 * 5. Print the result.
 *
 * Time Complexity: O(1), because the program performs a fixed number of
 * arithmetic operations, regardless of the input size.
 * Space Complexity: O(1), as it uses a few variables to store the dimensions
 * and area.
 */
#include <iostream>

using namespace std;

int main() {
  int m, n, area;
  // The area of a single 2x1 domino.
  int areaOneDominoes = 2;

  // Read the board dimensions.
  cin >> m >> n;
  // Calculate the total area of the board.
  area = m * n;
  
  // Calculate and print the maximum number of dominoes.
  cout << area / areaOneDominoes << endl;

  return 0;
}