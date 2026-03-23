/**
 * @file 231A.CPP
 * @brief Solution for Codeforces Problem 231A, "Team".
 * @details This program calculates the number of problems a team of three will solve.
 * A problem is solved if at least two members are confident. This implementation
 * reads all confidence values (n * 3) into a single array first, then processes
 * them in chunks of three to determine the number of solvable problems.
 *
 * @algorithm Counting with pre-loaded data
 * @complexity
 * Time: O(N), where N is the number of problems. The program iterates through all n*3 inputs twice.
 * Space: O(N), as it stores all n*3 inputs in an array.
 */
#include <iostream>

using namespace std;

int main() {
  // n: number of problems.
  // v: an array to store all confidence values. Size 3000 is sufficient as n <= 1000.
  // p: counter for solvable problems.
  int i, n, k, flag, p, v[3000];

  // Read the number of problems.
  cin >> n;

  // Read all n*3 confidence values into the array 'v'.
  for (i = 0; i < n * 3; i++)
    cin >> v[i];

  /**
   * @brief Main loop to iterate through problems and count solvable ones.
   * The outer loop iterates through the problems, taking 3 elements from 'v' at a time.
   * Pre-condition: 'v' is filled with all confidence scores.
   * Invariant: After each iteration of the outer loop, 'p' holds the count of
   * solvable problems among the problems processed so far.
   */
  for (i = 0, p = 0; i < n * 3; i += 3) {
    flag = 0;
    // Inner loop to sum the confidences for the current problem.
    for (k = i; k < i + 3; k++) {
      if (v[k] == 1)
        flag++;
    }
    // If two or more members are confident, the problem is solvable.
    if (flag >= 2)
      p++;
  }

  // Print the total count of solvable problems.
  cout << p;

  return 0; // Indicate successful execution.
}