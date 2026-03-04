/**
 * @file 231A.CPP
 * @brief Solution for Codeforces Problem 231A - "Team".
 *
 * This C-style program determines the number of problems a team will solve.
 * A problem is solved if at least two of the three members are confident
 * (represented by 1). The program reads 'n' sets of three opinions and
 * counts how many sets meet this criterion.
 *
 * @algorithm Simple counting by summation.
 * @time_complexity O(N), where N is the number of problems.
 * @space_complexity O(1).
 */
#include <cstdio>

using namespace std;

int main() {
  int x, y, z, count = 0, n;
  scanf("%d", &n);
  /**
   * @brief Processes each problem.
   * This loop iterates 'n' times. In each iteration, it reads the three
   * opinions (x, y, z). Since the opinions are either 0 or 1, their sum
   * equals the number of confident members. If this sum is 2 or more,
   * the problem is solved, and the counter is incremented.
   */
  while (n--) {
    scanf("%d %d %d", &x, &y, &z);
    if ((x + y + z) >= 2) {
      count++;
    }
  }
  printf("%d\n", count);
  return 0;
}