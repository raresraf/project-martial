/**
 * @00b3e221-25f1-47e1-926a-ac920a8ee93e/263A.CPP
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
  string a;          /**< Temporary string to read each row of the matrix. */
  string result;     /**< Stores the entire row (as a string) that contains '1' once located. */
  int n = 5;         /**< Loop counter for iterating through the 5 rows of the matrix. */
  int y = 0;         /**< Current 0-indexed row number being processed during matrix scan. */
  int cy = 0;        /**< Stores the 0-indexed row number where the digit '1' is eventually found. */

  /**
   * Block Logic: Reads the 5x5 matrix row by row and identifies the row
   *              containing the digit '1'.
   * Pre-condition: `n` is initialized to 5, `y` to 0.
   * Invariant: After this loop, `result` holds the row string where '1' was found,
   *            and `cy` holds its 0-indexed row number. `y` will be 5.
   */
  while (n--) {
    getline(cin, a); // Functional Utility: Reads an entire line from standard input, representing a row of the matrix.
    if (doesOneExist(a)) {
      cy = y;       // Functional Utility: Records the 0-indexed row number of the '1'.
      result = a;   // Functional Utility: Stores the specific row string that contains the '1'.
    }
    y++;            // Invariant: Increments the current row counter, ensuring progression through the matrix.
  }

  y = cy; /**< Functional Utility: Assigns the identified row of '1' to `y` for further calculations. */
  /**
   * Functional Utility: Calculates the minimum vertical moves needed to center the '1'.
   * This is achieved by computing the absolute difference between the '1''s row index
   * and the target center row index (2 for a 0-indexed 5x5 matrix). This represents
   * the vertical component of the Manhattan distance.
   */
  int moves = abs(2 - y); /**< Inline: Initializes `moves` with the vertical Manhattan distance. */

  vector<int> x; /**< Functional Utility: A dynamically sized array to store the integer (0 or 1) representation of each character from the `result` string, facilitating column-wise processing. */

  /**
   * Block Logic: Converts the `result` string (row containing '1') into a vector
   *              of integers (0 or 1), effectively preparing for column identification.
   * Pre-condition: `result` contains the row string identified in the previous `while` loop.
   * Invariant: `x` contains the integer representation of the characters in `result`.
   */
  for (char &c : result) {
    if (c == '0') {
      x.push_back(0); // Functional Utility: Appends 0 to vector 'x' if the character is '0'.
    } else if (c == '1') {
      x.push_back(1); // Functional Utility: Appends 1 to vector 'x' if the character is '1'.
    }
  }

  /**
   * Block Logic: Iterates through the vector `x` (representing the row containing '1')
   *              to precisely locate the column index of the digit '1' and calculate
   *              the necessary horizontal moves.
   * Pre-condition: `x` has been populated with the integer representation of the row containing '1'.
   * Invariant: `moves` accumulates the total Manhattan distance (vertical + horizontal).
   */
  for (int i = 0; i < x.size(); i++) {
    /**
     * Block Logic: Identifies the position of '1' within the current row.
     * Pre-condition: `x[i]` represents a cell in the matrix row.
     */
    if (x[i] == 1) {
      // Functional Utility: Adds the absolute difference between the '1''s column index (`i`)
      // and the target center column index (2) to the total `moves`. This represents
      // the horizontal component of the Manhattan distance.
      moves += abs(2 - i); // Inline: Accumulates the horizontal Manhattan distance.
    }
  }
  /**
   * Functional Utility: Outputs the total minimum number of moves required to reposition
   * the '1' to the center of the 5x5 matrix to standard output, followed by a newline.
   */
  cout << moves << endl;
}