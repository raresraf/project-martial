/*
 * Problem description
 *
 * https://codeforces.com/problemset/problem/50/A
 *
 */

/**
 * @file 50A.CPP
 * @brief Solution for Codeforces Problem 50A: Domino piling.
 * @details This program calculates the maximum number of 2x1 dominoes that can be placed
 * on an M x N rectangular board. Each domino covers exactly two squares. The solution
 * leverages the fact that each domino covers two squares, so the maximum number of dominoes
 * is simply half the total area of the board.
 * @algorithm Mathematical formula (Area / 2).
 * Time Complexity: $O(1)$ as it involves a simple arithmetic calculation.
 * Space Complexity: $O(1)$ to store a few variables.
 */
#include <bits/stdc++.h>
using namespace std;

/*
 * Problem description
 *
 * https://codeforces.com/problemset/problem/50/A
 *
 */

/**
 * @file 50A.CPP
 * @brief Solution for Codeforces Problem 50A: Domino piling.
 * @details This program calculates the maximum number of 2x1 dominoes that can be placed
 * on an M x N rectangular board. Each domino covers exactly two squares. The solution
 * leverages the fact that each domino covers two squares, so the maximum number of dominoes
 * is simply half the total area of the board.
 * @algorithm Mathematical formula (Area / 2).
 * Time Complexity: $O(1)$ as it involves a simple arithmetic calculation.
 * Space Complexity: $O(1)$ to store a few variables.
 */
#include <bits/stdc++.h> // Includes most standard C++ library headers (e.g., iostream for cin/cout).
using namespace std;     // Uses the standard namespace.

int main() {
  /**
   * @brief Main function: orchestrates the domino piling calculation.
   */
  int m, // @var m: The number of rows of the rectangular board.
      n; // @var n: The number of columns of the rectangular board.

  // Functional Utility: Reads the dimensions of the board (m and n).
  cin >> m >> n;

  /**
   * @brief Calculates the maximum number of dominoes.
   * @details Since each domino covers 2 squares, the maximum number of dominoes that can be placed
   *          on an M x N board is simply the total area (M * N) divided by 2. Integer division
   *          automatically handles cases where the area is odd (the remaining single square cannot
   *          be covered by a 2x1 domino).
   * @invariant `(n * m) / 2` correctly represents the maximum number of dominoes.
   */
  cout << (n * m) / 2 << endl; // Prints the result.
  return 0; // Indicates successful execution.
}
