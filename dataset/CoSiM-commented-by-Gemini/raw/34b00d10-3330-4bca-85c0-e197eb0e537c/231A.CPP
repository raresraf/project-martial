/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem.
 *
 * This program determines how many problems a team of three will solve.
 * It reads 'n', the number of problems. For each problem, it reads the
 * opinions of three members (1 if they are sure, 0 otherwise). The team
 * solves a problem only if at least two members are sure. The program
 * outputs the total number of problems that will be solved.
 *
 * Algorithm:
 * 1. Read the number of problems, n.
 * 2. Read the 3 opinions for each of the n problems into a 2D array.
 * 3. Initialize a counter for solved problems to 0.
 * 4. Iterate through each problem (each row of the array):
 *    a. Sum the opinions (number of 1s) for the current problem.
 *    b. If the sum is 2 or more, increment the solved problems counter.
 * 5. Print the final count.
 *
 * Time Complexity: O(N), where N is the number of problems. Although there are
 * nested loops, the inner loop runs a fixed 3 times, so the complexity is linear
 * with respect to the number of problems.
 * Space Complexity: O(N) to store the opinions for all problems in an array.
 */
#include <iostream>
using namespace std;
int main() {
  // i, j: loop counters.
  // n: number of problems.
  // o: temporary counter for 'sure' votes for a single problem.
  // p: total count of problems that will be solved.
  int i, j, n, m, o, p; // 'm' is declared but not used.
  int a[1001][4]; // Array to store the votes for each problem.

  // Read the total number of problems.
  cin >> n;

  /**
   * Block Logic: Read all input data.
   * This nested loop populates the 2D array 'a' with the votes
   * (0 or 1) for each of the 'n' problems.
   */
  for (i = 0; i < n; i++) {
    for (j = 0; j < 3; j++) {
      cin >> a[i][j];
    }
  }
  
  p = 0; // Initialize the count of solvable problems.
  
  /**
   * Block Logic: Process each problem to see if it's solvable.
   */
  for (i = 0; i < n; i++) {
    o = 0; // Reset the 'sure' vote counter for the current problem.
    /**
     * Block Logic: Count the 'sure' votes for one problem.
     */
    for (j = 0; j < 3; j++) {
      if (a[i][j] == 1) {
        o++;
      }
    }
    // Pre-condition: Check if the number of 'sure' votes is 2 or more.
    if (o > 1) {
      p++; // If so, increment the total count of solvable problems.
    }
  }

  // Print the final result.
  cout << p << endl;
  return 0;
}
