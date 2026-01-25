/**
 * @file 263A.CPP
 * @brief Calculates the Manhattan distance of a '1' from the center of a 5x5 matrix.
 *
 * This program reads a 5x5 matrix of 0s and a single 1. It identifies the coordinates
 * of the '1' and calculates the number of single row or column moves (Manhattan distance)
 * required to move it to the center of the matrix, at position (3,3) (or 1-indexed).
 * The center is at (2,2) in 0-indexed coordinates.
 *
 * @b Algorithm: The program iterates through the rows of the matrix to find the row
 * containing the '1'. It then determines the column of the '1' within that row.
 * The total number of moves is the sum of the absolute difference in rows from the
 * center row and the absolute difference in columns from the center column.
 *
 * @b Time_Complexity: O(1), as the input is always a 5x5 matrix.
 * @b Space_Complexity: O(1), as it uses a fixed amount of storage.
 */
#include <iostream>
#include <string>
#include <vector>

using namespace std;

/**
 * @brief Checks if the character '1' exists within a given string.
 * @param c The input string.
 * @return True if '1' is found, otherwise false.
 */
bool doesOneExist(string c) {
  if (c.find('1') != std::string::npos) {
    return true;
  }
  return false;
}

int main() {
  string a;
  string result;
  int n = 5, y = 0, cy = 0;

  /**
   * @brief This loop reads the 5x5 matrix row by row to find the row containing '1'.
   *
   * Pre-condition: Input stream contains 5 lines representing the matrix rows.
   * Invariant: `y` tracks the current row index (0-indexed). `cy` stores the row index
   * where '1' is found, and `result` holds the content of that row string.
   */
  while (n--) {
    getline(cin, a);
    if (doesOneExist(a)) {
      cy = y;
      result = a;
    }
    y++;
  }

  y = cy;
  // Calculate the vertical distance (number of row moves) from the center row (index 2).
  int moves = abs(2 - y);

  vector<int> x;
  // This loop parses the characters of the row string into a vector of integers,
  // effectively ignoring the spaces.
  for (char &c : result) {
    if (c == '0') {
      x.push_back(0);
    } else if (c == '1') {
      x.push_back(1);
    }
  }

  /**
   * @brief This loop finds the column of the '1' in the parsed vector and adds the horizontal
   * distance to the total moves. The center column is at index 2.
   */
  for (int i = 0; i < x.size(); i++) {
    if (x[i] == 1)
      // The horizontal distance is calculated from the center column (index 2).
      moves += abs(2 - i);
  }
  cout << moves << endl;
}
