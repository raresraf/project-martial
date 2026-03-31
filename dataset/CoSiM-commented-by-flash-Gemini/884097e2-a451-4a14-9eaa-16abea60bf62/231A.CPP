/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem from Codeforces (Problem 231A).
 * This program counts the number of problems a team of three friends will implement.
 * A problem is implemented if at least two friends are sure about the solution.
 *
 * Algorithm: Iterative counting. For each problem, the confidence levels of the three
 * friends are read. If the sum of confident friends is 2 or more, a counter for
 * implementable problems is incremented.
 * Time Complexity: O(N), where N is the number of problems, as each problem and
 *                  its three confidence values are processed once.
 * Space Complexity: O(1) as only a few variables are used.
 */

#include <bits/stdc++.h> // @brief Includes many standard libraries (e.g., iostream, string, algorithm).

using namespace std;

int main() {
  /**
   * @brief Main function to determine the number of problems a team will implement.
   * Reads the total number of problems, then for each problem, reads the confidence
   * of three friends, and increments a counter if at least two friends are confident.
   */
  int t, a, count1 = 0, count2 = 0; // @brief t: number of problems; a: friend's confidence; count1: confident friends for current problem; count2: total implementable problems.
  scanf("%d", &t); // @brief Read the total number of problems (t).
  
  // Block Logic: Iterate through each problem.
  // Pre-condition: 't' problems are expected to be processed.
  // Invariant: 'count2' accurately reflects the number of implementable problems encountered so far.
  for (int i = 1; i <= t; i++) {
    // Block Logic: Read confidence levels for the three friends for the current problem.
    // 'j' iterates 3 times for 3 friends.
    for (int j = 1; j <= 3; j++) {
      scanf("%d", &a); // Inline: Read friend's confidence (1 for sure, 0 for not sure).
      if (a == 1) {
        count1++; // Inline: Increment count of confident friends for this problem.
      }
    }
    // Block Logic: Determine if the team will implement the current problem.
    // If at least two friends are confident (`count1 >= 2`).
    if (count1 >= 2) {
      count2++; // Inline: Increment the total count of implementable problems.
    }
    count1 = 0; // @brief Reset confident friend counter for the next problem.
  }
  printf("%d\n", count2); // @brief Print the total number of problems the team will implement.

  return 0; // @brief Indicates successful program termination.
}
