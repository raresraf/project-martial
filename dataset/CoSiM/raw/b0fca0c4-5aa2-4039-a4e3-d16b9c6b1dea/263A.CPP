#include <cmath>
#include <iostream>

/*
 * Contest: Code Forces Round #161 (Task 263-A)
 * URL: http://codeforces.ru/contest/263/problem/A
 */

int main(int argc, char **argv) {

  int row, col, M[5][5];

  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      std::cin >> M[i][j];

      if (M[i][j]) {
        row = i;
        col = j;
      }
    }
  }

  std::cout << (std::abs(row - 2) + std::abs(col - 2)) << std::endl;
  return 0;
}
