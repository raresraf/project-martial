/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem from Codeforces contest 231.
 *
 * @details
 * The problem is about determining how many problems a team of three will decide to implement.
 * The team will only implement a problem if at least two of its three members are sure about
 * the solution. The input consists of `n` problems, and for each problem, three boolean values
 * (represented as 0 or 1) indicating whether each member is sure.
 *
 * The program reads the number of problems `n`. Then, it iterates `n` times. In each iteration,
 * it reads the three members' opinions. It counts how many members are sure (i.e., their input is 1).
 * If this count is two or greater, a counter for the total number of problems to be implemented
 * is incremented. Finally, the total count is printed.
 *
 * Algorithm:
 * 1. Read the total number of problems, `n`.
 * 2. Initialize a counter for implementable problems to zero.
 * 3. Loop `n` times:
 *    a. Read the three integer opinions (0 or 1) for the current problem.
 *    b. Sum the three integers to get the number of members who are sure.
 *    c. If the sum is 2 or greater, increment the main counter.
 * 4. Print the final value of the counter.
 *
 * Time Complexity: O(n), where `n` is the number of problems, as we iterate through each problem once.
 * Space Complexity: O(1), as we only use a few variables to store the counts and inputs for each problem.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {

  int n;
  // Read the number of problems.
  cin >> n;

  int counter = 0; // Counter for problems that will be implemented.
  int smallcounter = 0; // Counter for members sure about a single problem.

  // Block Logic: Iterate through each of the 'n' problems.
  for (int i = 0; i < n; i++) {

    smallcounter = 0; // Reset the count for each new problem.
    int a, b, c;
    // Read the opinions of the three team members for the current problem.
    cin >> a >> b >> c;

    // Block Logic: Count how many members are sure about the solution.
    if (a == 1) {
      smallcounter++;
    }

    if (b == 1) {
      smallcounter++;
    }

    if (c == 1) {
      smallcounter++;
    }

    // Pre-condition: Check if at least two members are sure.
    if (smallcounter >= 2) {
      // If so, increment the counter for problems to be implemented.
      counter++;
    }
  }

  // Output the total number of problems the team will implement.
  cout << counter;
}
