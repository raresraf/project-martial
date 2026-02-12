/**
 * @file 50A.CPP
 * @brief Solves the "Domino piling" problem from Codeforces.
 * @details The problem asks for the maximum number of 2x1 dominoes that can be placed on a
 * rectangular board of size M x N. The total area of the board is M * N. Since each
 * domino covers an area of 2, the maximum number of non-overlapping dominoes is the
 * total board area divided by 2, using integer division.
 *
 * @author Taro Young
 * @date 2020/9/7
 *
 * @algorithm Area calculation.
 * @complexity
 * Time complexity: O(1), as it involves a single arithmetic operation.
 * Space complexity: O(1).
 */
#include <iostream>

using namespace std;

int main() {
  // m, n: the dimensions of the rectangular board.
  int m, n;
  cin >> m >> n;
  // The total number of 2x1 squares is the total area divided by 2.
  cout << m * n / 2;
  return 0;
}
