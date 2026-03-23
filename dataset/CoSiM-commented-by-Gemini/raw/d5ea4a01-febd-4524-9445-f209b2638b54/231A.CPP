/**
 * @file 231A.CPP
 * @brief Solution for Codeforces Problem 231A, "Team".
 * @details This program counts the number of problems a team of three will solve.
 * A problem is solved if at least two members are confident (represented by '1').
 * This particular implementation reads the confidence levels as strings and compares
 * them to "1" to count the number of confident members for each problem.
 *
 * @algorithm Simple Counting
 * @complexity
 * Time: O(N), where N is the number of problems. The loop runs N times.
 * Space: O(1), as it uses a fixed amount of memory per iteration. (The strings are small and overwritten).
 */
// CodeForces
// Team
// GNU C++17

#include <iostream>
#include <string>
// The following headers were included in the original file but are not used.
// #include <algorithm>
// #include <iterator>
// #include <sstream>
// #include <vector>

using namespace std;

int main() {
  int numberOfTests = 0;
  int validProblems = 0; // Counter for problems that will be solved.
  int count = 0;         // Counter for confident members for a single problem.
  
  cin >> numberOfTests;

  /**
   * @brief Main loop to process each problem.
   * Pre-condition: 'numberOfTests' holds the total number of problems.
   */
  while (numberOfTests > 0) {
    // Reading opinions as strings, which is less efficient than integers but works.
    string guess, guessTwo, guessThree;
    cin >> guess >> guessTwo >> guessThree;

    // Count how many members are confident.
    if (guess == "1")
      count++;
    if (guessTwo == "1")
      count++;
    if (guessThree == "1")
      count++;

    // If two or more members are confident, the problem is valid.
    if (count >= 2)
      validProblems++;

    numberOfTests--;
    // Reset the counter for the next problem.
    count = 0;
  }

  // Print the total number of solvable problems.
  cout << validProblems << endl;
  return 0; // Indicate successful execution.
}