/**
 * @file 231A.CPP
 * @brief Solution for a counting problem based on a team's consensus (Codeforces 231A).
 * @details This program calculates the number of problems that will be implemented
 * by a team of three. A problem is implemented if at least two of the three
 * members are confident in the solution.
 *
 * @section
 *
 * Algorithm:
 * The program iterates through a given number of problems, `n`. For each problem,
 * it reads the confidence levels (0 or 1) of the three members. The sum of these
 * levels is calculated and stored. A ternary operator then checks if this sum is
 * two or greater. If the condition is met, a counter (`ans`) is incremented.
 * After checking all problems, the final count is printed.
 *
 * - Time Complexity: O(N), where N is the number of problems.
 * - Space Complexity: O(1).
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  int n;
  int ans = 0;
  cin >> n;
  /**
   * Block Logic: Iterates 'n' times, processing one problem in each iteration.
   * Invariant: Before each iteration, 'ans' holds the count of problems solved
   *            from the set of problems already processed.
   */
  for (int i = 0; i < n; ++i) {
    int petya, vasya, tonya;
    cin >> petya >> vasya >> tonya;
    int confidence = petya + vasya + tonya;
    /**
     * Inline Logic: Uses a ternary operator to increment the answer if the
     * sum of confidence levels is 2 or more. The 'else' clause is redundant
     * but does not affect the outcome.
     */
    (confidence >= 2) ? ++ans : ans = ans;
  }
  cout << ans;
}
