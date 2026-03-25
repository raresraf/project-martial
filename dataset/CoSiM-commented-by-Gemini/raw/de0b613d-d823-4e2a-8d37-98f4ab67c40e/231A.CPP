// 231A - Team

#include <iostream>

/**
 * @file 231A.CPP
 * @brief A concise solution for the "Team" problem.
 *
 * Problem Statement:
 * A team of three members participates in a programming contest. For each
 * problem, we know which members are confident about the solution. The team
 * will only implement a problem if at least two members are confident. Given
 * the confidence status for `n` problems, count how many problems the team
 * will implement.
 *
 * Algorithm:
 * This solution uses a direct and efficient approach.
 * 1. Read the total number of problems, `n`.
 * 2. Loop `n` times, once for each problem.
 * 3. In each iteration, read the three confidence values (0 or 1) for the
 *    members.
 * 4. Sum these three values. The sum directly corresponds to the number of
 *    confident members.
 * 5. If the sum is 2 or greater, increment a counter for solved problems.
 * 6. After the loop, print the total count.
 *
 * Time Complexity: O(N), where N is the number of problems.
 * Space Complexity: O(1).
 */

using namespace std;

int main() {
  // n: number of problems
  // p, v, t: confidence of Petya, Vasya, and Tonya respectively (1 or 0)
  // sum: counter for the number of problems the team will solve
  int n, p, v, t, sum = 0;

  cin >> n;

  // Loop n times, decrementing n in each iteration.
  while (n--) {
    cin >> p >> v >> t;

    // Pre-condition: Check if at least two members are sure.
    // Summing the inputs (0 or 1) gives the count of confident members.
    if (p + v + t >= 2) {
      sum++;
    }
  }

  cout << sum;

  return 0;
}