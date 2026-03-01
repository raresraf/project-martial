/**
 * @file 50A.CPP
 * @brief Solves the Domino Piling problem.
 *
 * @problem Given a rectangular board of size M x N, find the maximum number
 * of 2x1 dominoes that can be placed on it.
 *
 * @algorithm The total area of the board is M * N. Each domino covers an area of 2.
 * Therefore, the maximum number of dominoes that can be placed is the total area
 * divided by 2, rounded down to the nearest integer. This is equivalent to
 * integer division: `(m * n) / 2`.
 * The code implements this logic using a case analysis based on the parity of
 * M and N, but all branches mathematically simplify to the same result.
 *
 * @time_complexity O(1), as it involves a constant number of arithmetic operations.
 * @space_complexity O(1).
 */
#include <cmath>
#include <iostream>
using namespace std;

int main() {
  int m, n;
  cin >> m >> n;

  // The code calculates the result based on the parity of m and n.
  // However, all three cases produce the same result as integer division (m*n)/2.
  if (m % 2 == 0 && n % 2 == 0) {
    // Case 1: Both dimensions are even. The area is a multiple of 2.
    cout << (m * n) / 2;
  } else if (m % 2 != 0 && n % 2 != 0) {
    // Case 2: Both dimensions are odd. The area is odd.
    // The expression ((m-1)*(n-1))/2 + (m+n-2)/2 simplifies to (m*n-1)/2,
    // which is the correct integer division result.
    cout << ((m - 1) * (n - 1)) / 2 + (m + n - 2) / 2;
  } else {
    // Case 3: One dimension is even, the other is odd. The area is a multiple of 2.
    cout << (m * n) / 2;
  }
  return 0;
}
