/**
 * @file 231A.CPP
 * @brief Solution for Codeforces Problem 231A - "Team".
 * This program determines the number of problems a team of three friends will implement
 * in a programming contest. A problem is implemented if at least two friends are
 * sure about its solution.
 * Algorithm: Iterates through problems, sums certainty votes, and counts problems with >= 2 votes.
 * Time Complexity: O(N), where N is the number of problems.
 * Space Complexity: O(1) - uses a constant amount of memory for variables.
 */
#include <cstdio> // Required for standard C-style input/output functions (scanf, printf)

using namespace std; // Uses the standard namespace

int main() {
  // x, y, z: represent the certainty of each of the three friends (1 if sure, 0 if not).
  // count: accumulates the number of problems that will be implemented.
  // n: stores the total number of problems.
  int x, y, z, count = 0, n;
  
  // Reads the total number of problems from standard input.
  scanf("%d", &n);
  
  /**
   * Block Logic: Processes each problem one by one, determining if it will be implemented.
   * The loop runs 'n' times, decrementing 'n' in each iteration.
   * Invariant: `count` holds the number of problems implemented so far.
   */
  while (n--) {
    // Reads the certainty of the three friends for the current problem.
    scanf("%d %d %d", &x, &y, &z);
    
    // Checks if the sum of certainties is 2 or more, meaning at least two friends are sure.
    if ((x + y + z) >= 2) {
      count++; // If so, increment the count of implemented problems.
    }
  }
  
  // Prints the total number of problems that will be implemented.
  printf("%d\n", count);
  
  return 0; // Indicates successful program execution.
}
