/**
 * @file 263A.CPP
 * @brief Solution for the "Beautiful Matrix" problem from Codeforces.
 *
 * This problem involves a 5x5 matrix where exactly one cell contains the
 * integer '1' and all other cells contain '0'. The goal is to find the minimum
 * number of moves required to bring the '1' to the central cell of the matrix
 * (row 2, column 2, assuming 0-indexed rows/columns or row 3, column 3, 1-indexed).
 * A move consists of swapping any adjacent rows or adjacent columns.
 *
 * Problem Link: https://codeforces.com/problemset/problem/263/A
 *
 * Algorithm: Matrix scanning to locate the '1', followed by Manhattan distance
 *            calculation from its current position to the center (2,2).
 * Time Complexity: O(R * C), where R and C are the dimensions of the matrix.
 *                  Since R=5 and C=5, this is effectively O(1).
 * Space Complexity: O(C) for storing a row and extracting column data.
 *                   Effectively O(1) for a 5x5 matrix.
 */

#include <iostream>
#include <string>
#include <vector>
#include <cmath> // For abs()

using namespace std;

/**
 * @brief Checks if the character '1' exists within a given string (row of the matrix).
 *
 * Functional Utility: Used to quickly identify the row that contains the '1'
 * in the Beautiful Matrix problem.
 *
 * @param c The input string representing a row of the matrix.
 * @return True if '1' is found in the string, False otherwise.
 */
bool doesOneExist(string c) {
  // `string::npos` is returned by `find` if the substring is not found.
  if (c.find('1') != std::string::npos) {
    return true;
  }
  return false;
}

int main() {
  string a;          // Temporary string to read each row of the matrix.
  string result;     // Stores the entire row (as a string) that contains '1'.
  int n = 5;         // Loop counter for 5 rows.
  int y = 0;         // Current 0-indexed row number being processed.
  int cy = 0;        // Stores the 0-indexed row number where '1' is found.

  /**
   * Block Logic: Reads the 5x5 matrix row by row and identifies the row
   *              containing the digit '1'.
   * Invariant: After this loop, `result` holds the row string where '1' was found,
   *            and `cy` holds its 0-indexed row number.
   */
  while (n--) {
    getline(cin, a); // Reads an entire line (row) of the matrix.
    if (doesOneExist(a)) {
      cy = y;       // Store the row index where '1' is present.
      result = a;   // Store the actual row string.
    }
    y++;            // Increment current row counter.
  }

  y = cy; // `y` now represents the 0-indexed row of '1'.
  // Calculates the minimum vertical moves required to bring the '1' to the center row (index 2).
  int moves = abs(2 - y); // Manhattan distance for rows.

  vector<int> x; // Vector to store integer representation of the characters in the `result` string.

  /**
   * Block Logic: Converts the `result` string (row containing '1') into a vector
   *              of integers (0 or 1) and then finds the column index of '1'.
   * Invariant: `x` contains the integer representation of the characters in `result`.
   */
  for (char &c : result) {
    if (c == '0') {
      x.push_back(0);
    } else if (c == '1') {
      x.push_back(1);
    }
  }

  /**
   * Block Logic: Iterates through the vector representing the row with '1'
   *              to find its column index and calculate horizontal moves.
   * Invariant: `moves` accumulates the total Manhattan distance (vertical + horizontal).
   */
  for (int i = 0; i < x.size(); i++) {
    if (x[i] == 1) {
      // Calculates the minimum horizontal moves required to bring the '1' to the center column (index 2).
      moves += abs(2 - i); // Manhattan distance for columns.
    }
  }
  // Prints the total minimum number of moves.
  cout << moves << endl;
}