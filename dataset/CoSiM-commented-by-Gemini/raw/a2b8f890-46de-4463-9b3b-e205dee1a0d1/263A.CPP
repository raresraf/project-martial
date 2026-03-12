/**
 * @file 263A.CPP
 * @brief Solves the "Beautiful Matrix" problem by simulating row and column swaps.
 *
 * The program calculates the minimum number of moves to make a 5x5 matrix "beautiful"
 * by moving the single digit '1' to the center of the matrix (position 2,2).
 * A move is defined as swapping two adjacent rows or two adjacent columns.
 * The number of moves is equivalent to the Manhattan distance from the '1' to the center.
 *
 * @algorithm Brute-force simulation of swaps.
 * @time_complexity O(1), as the matrix size is fixed at 5x5. The loops run a constant number of times.
 * @space_complexity O(1) for the 5x5 matrix.
 * @note The implemented solution physically simulates the swaps, which is functionally correct
 * but more complex than directly calculating the Manhattan distance.
 */
#include <iostream>
#include <string>

using namespace std;

int main() {
  int counter = 0;
  int row = 2;
  int coloum = 2;
  string matrix[5][5];
  /**
   * @brief Reads the 5x5 matrix from input and finds the coordinates of the '1'.
   * The row and coloum variables are updated to the position of '1'.
   */
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      char temp;
      cin >> temp;
      matrix[i][j] = temp;
      if (temp == '1') {
        row = i;
        coloum = j;
      }
    }
  }

  /**
   * @brief Simulates moving the row containing '1' to the center row (index 2).
   * Each swap of adjacent rows moves the '1' one step closer to the target row
   * and increments the move counter.
   */
  while (row != 2) {
    if (row < 2) {
      // This swaps the current row with the one below it.
      for (int i = 0; i < 5; i++) {
        string temp = matrix[row][i];
        matrix[row][i] = matrix[row + 1][i];
        matrix[row + 1][i] = temp;
      }
      row++;
    } else if (row > 2) {
      // This swaps the current row with the one above it.
      for (int i = 0; i < 5; i++) {
        string temp = matrix[row][i];
        matrix[row][i] = matrix[row - 1][i];
        matrix[row - 1][i] = temp;
      }
      row--;
    }
    counter++;
  }

  /**
   * @brief Simulates moving the column containing '1' to the center column (index 2).
   * Each swap of adjacent columns moves the '1' one step closer to the target column
   * and increments the move counter.
   */
  while (coloum != 2) {
    if (coloum < 2) {
      // This swaps the current column with the one to its right.
      for (int i = 0; i < 5; i++) {
        string temp = matrix[i][coloum];
        matrix[i][coloum] = matrix[i][coloum + 1];
        matrix[i][coloum + 1] = temp;
      }
      coloum++;
    } else if (coloum > 2) {
      // This swaps the current column with the one to its left.
      for (int i = 0; i < 5; i++) {
        string temp = matrix[i][coloum];
        matrix[i][coloum] = matrix[i][coloum - 1];
        matrix[i][coloum - 1] = temp;
      }
      coloum--;
    }
    counter++;
  }
  cout << counter;
  return 0;
}