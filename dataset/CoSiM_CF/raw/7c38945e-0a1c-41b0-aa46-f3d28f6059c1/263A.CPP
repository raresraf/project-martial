#include <iostream>

int main() {
  int matrix[5][5];
  int pl_row, pl_col;
  for (int row = 0; row < 5; row++) {
    for (int col = 0; col < 5; col++) {
      std::cin >> matrix[row][col];
      if (matrix[row][col] == 1) {
        pl_row = row;
        pl_col = col;
      }
    }
  }
  std::cout << std::abs(pl_row - 2) + std::abs(pl_col - 2) << std::endl;

  return 0;
}
