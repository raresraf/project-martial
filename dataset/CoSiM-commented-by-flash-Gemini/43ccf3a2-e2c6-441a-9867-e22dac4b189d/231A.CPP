/**
 * @file 231A.CPP
 * @brief Solution to a competitive programming problem (Codeforces 231A - "Team").
 *
 * This program calculates the number of problems a team of three friends will implement.
 * The rule for implementation is that a problem is implemented if at least two
 * of the three friends are sure about its solution.
 *
 * Algorithm: Simple counting based on a conditional sum.
 * The program iterates through 'n' problems. For each problem, it reads three
 * binary inputs (0 or 1, representing a friend's certainty). If the sum of these
 * inputs is greater than 1 (meaning at least two friends are sure), a counter
 * for implemented problems is incremented.
 *
 * Time Complexity: O(N), where N is the number of problems, due to a single loop.
 * Space Complexity: O(1), as only a few variables are used irrespective of N.
 */
#include <iostream> // Required for input/output operations (cin, cout).
using namespace std;

/**
 * @brief Main function: Entry point of the program.
 *
 * Reads the number of problems and for each problem, determines if it will be
 * implemented by the team, then prints the total count of implemented problems.
 */
int main() {
  int n;               // Variable to store the total number of problems.
  int counter = 0;     // Counter for problems that will be implemented.
  int a, b, c;         // Variables to store the opinions of the three friends.

  // Pre-condition: Read 'n', the total number of problems.
  cin >> n;
  // Block Logic: Iterate 'n' times, once for each problem.
  // Invariant: 'n' decreases with each iteration, loop continues until all problems are processed.
  while (n--) {
    // Pre-condition: Read the opinions of the three friends (a, b, c) for the current problem.
    // (1 indicates 'sure', 0 indicates 'not sure').
    cin >> a >> b >> c;
    // Block Logic: Check if at least two friends are sure about the solution.
    // If the sum of their opinions is greater than 1, it means 2 or 3 friends are sure.
    if (a + b + c > 1) {
      counter++; // Increment the counter for problems that will be implemented.
    }
  }
  // Output the total number of problems that will be implemented, followed by a newline.
  cout << counter << endl;
}
