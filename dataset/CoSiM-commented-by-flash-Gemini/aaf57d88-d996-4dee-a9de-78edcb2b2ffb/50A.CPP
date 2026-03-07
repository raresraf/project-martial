/**
 * @file 50A.CPP
 * @brief Solution to the "Domino piling" competitive programming problem (Codeforces 50A).
 * The problem asks to find the maximum number of 2x1 dominoes that can be placed
 * on a rectangular M x N (here L x B) board without overlapping, where each domino
 * covers exactly two squares.
 * Algorithm: Area-based calculation. Each domino covers 2 squares. The total area
 * is L * B. The maximum number of dominoes is (L * B) / 2.
 * Time Complexity: O(1).
 * Space Complexity: O(1).
 */
#include <cstdio> // For standard input/output functions like `scanf` and `printf`.

using namespace std;

int main() {
  // Declares 'l' for the length and 'b' for the breadth (width) of the rectangular board.
  int l, b;
  // Reads the length and breadth of the board from standard input.
  scanf("%d %d", &l, &b);
  // Calculates the maximum number of dominoes and prints the result, followed by a newline.
  // The total area of the board is l * b. Since each domino covers 2 squares,
  // the maximum number of dominoes is (l * b) / 2.
  printf("%d\n", (l * b) / 2);
  
  return 0; // Indicates successful program execution.
}