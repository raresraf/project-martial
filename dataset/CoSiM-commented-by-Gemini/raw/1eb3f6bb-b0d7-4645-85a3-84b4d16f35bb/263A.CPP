/**
 * @file 263A.CPP
 * @brief Solution for Codeforces problem 263A, "Beautiful Matrix".
 *
 * @description
 * The problem involves a 5x5 matrix containing a single '1' and 24 '0's.
 * The goal is to find the minimum number of moves (swapping adjacent rows or
 * columns) required to move the '1' to the center of the matrix, which is at
 * position (3, 3).
 *
 * The minimum number of moves is equivalent to the Manhattan distance between
 * the current position of the '1' and the center (3, 3).
 * Manhattan distance = |x1 - x2| + |y1 - y2|.
 *
 * @algorithm
 * 1. Define a function to read the 5x5 matrix and find the coordinates of the '1'.
 * 2. In the main function, call this helper function to get the row and column
 *    of the '1'.
 * 3. Calculate the Manhattan distance from the found coordinates to (3, 3).
 * 4. Print the result.
 *
 * @time_complexity O(1), as the program always processes a fixed 5x5 grid (25 inputs).
 * @space_complexity O(1).
 */
#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Reads a 5x5 matrix from standard input to find the 1-indexed
 * coordinates of the number 1.
 * @param a Reference to an integer to store the row coordinate of '1'.
 * @param b Reference to an integer to store the column coordinate of '1'.
 */
void yoyo(int &a, int &b) {
  // Block Logic: Iterate through the 5x5 grid to find the '1'.
  for (int i = 1; i <= 5; i++) {
    for (int j = 1; j <= 5; j++) {
      int temp;
      cin >> temp;
      if (temp == 1) {
        a = i;
        b = j;
        return; // Exit as soon as '1' is found.
      }
    }
  }
}

int main() {
  // a: row of the '1', b: column of the '1'.
  int a, b;
  // Find the location of the '1' and store it in a and b.
  yoyo(a, b);
  // Calculate the Manhattan distance from (a, b) to the center (3, 3)
  // and print the result.
  cout << abs(3 - a) + abs(3 - b);
}
