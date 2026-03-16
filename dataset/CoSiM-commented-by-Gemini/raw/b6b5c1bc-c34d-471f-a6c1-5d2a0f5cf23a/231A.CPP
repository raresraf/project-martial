/**
 * @file 231A.CPP
 * @brief Solution to the "Team" problem (Codeforces 231A).
 *
 * @details
 * This program calculates the number of problems a team of three will decide to solve.
 * The team decides to solve a problem only if at least two of its three members are
 * confident about the solution.
 *
 * The program first reads the total number of problems, `n`. Then, for each problem,
 * it reads three integer inputs (0 or 1), where 1 signifies confidence and 0 signifies
 * a lack of confidence from a team member.
 *
 * The core logic relies on the fact that summing the three inputs gives the total number
 * of confident members. If this sum is 2 or greater, the team will solve the problem,
 * and a counter is incremented.
 *
 * Algorithm:
 * 1. Read the number of problems, `n`.
 * 2. Initialize a counter for solvable problems to 0.
 * 3. Loop `n` times:
 *    a. Read the three members' confidence values (a, b, c).
 *    b. If the sum `a + b + c` is 2 or more, increment the counter.
 * 4. Print the final counter.
 *
 * Time Complexity: O(n), as the program iterates through each of the `n` problems once.
 * Space Complexity: O(1), as a constant amount of memory is used regardless of input size.
 */
#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tgmath.h>

using namespace std;

int main(void) {
  int n;

  cin >> n;

  int counter = 0;

  /**
   * @brief Main loop to process each of the `n` problems.
   */
  for (int i = 0; i < n; i++) {
    int a, b, c;
    cin >> a >> b >> c;
    /**
     * @brief Checks if at least two team members are confident.
     * @details Since confident members are represented by 1 and non-confident by 0,
     * summing the three inputs directly gives the count of confident members.
     * The problem is solvable if this sum is 2 or 3.
     */
    if ((a + b + c) >= 2)
      counter++;
  }

  cout << counter;
  return 0;
}