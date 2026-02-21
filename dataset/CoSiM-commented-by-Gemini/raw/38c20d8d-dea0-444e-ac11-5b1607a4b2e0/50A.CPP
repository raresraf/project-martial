/**
 * @file 50A.CPP
 * @brief Solution to the "Domino piling" problem (Codeforces 50A).
 *
 * @details This program calculates the maximum number of 2x1 dominoes that can be
 * placed on a rectangular board of size M x N.
 *
 * @algorithm The problem can be solved by considering the total area of the board.
 * The board has an area of `M * N`. Each domino has an area of 2. Therefore,
 * the maximum number of non-overlapping dominoes that can fit on the board is
 * the total area divided by 2. Integer division `(M * N) / 2` correctly
 * computes `floor((M * N) / 2)`, which is the solution.
 *
 * @complexity
 * Time Complexity: O(1), the solution involves a single arithmetic calculation.
 * Space Complexity: O(1), as only a few variables are needed to store the dimensions.
 */
#include <cstdio>
using namespace std;

int main() {
  int n, m;
  scanf("%d%d", &m, &n);
  
  // Functional Logic: Calculate and print the maximum number of dominoes.
  printf("%d", (n * m) / 2);
}
