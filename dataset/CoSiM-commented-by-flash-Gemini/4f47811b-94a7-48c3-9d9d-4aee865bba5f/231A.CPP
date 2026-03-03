/**
 * @file 231A.CPP
 * @brief Solution for Codeforces Problem 231A - "Team".
 * This program determines the number of problems a team of three friends will implement
 * in a programming contest. A problem is implemented if at least two friends are
 * sure about its solution.
 * Algorithm: Iterates through problems, sums certainty votes, and counts problems with >= 2 votes.
 * Time Complexity: O(N), where N is the number of problems.
 * Space Complexity: O(1) - uses a constant amount of memory for variables and a fixed-size array.
 */
#include <iostream> // Required for standard input/output operations (cin, cout).

using namespace std; // Uses the standard namespace.

int main() {
  int n = 0; // Declares an integer to store the total number of problems, initialized to 0.
  cin >> n;  // Reads the total number of problems from standard input.
  
  int counter = 0; // Initializes a counter for problems that will be implemented.

  /**
   * Block Logic: Processes each problem one by one, determining if it will be implemented.
   * The loop runs 'n' times, once for each problem.
   * Invariant: `counter` holds the number of problems implemented so far.
   */
  for (int i = 0; i < n; i++) {
    int nums[3]; // Declares an array to store the certainty votes (0 or 1) of the three friends.

    // Reads the certainty votes of the three friends for the current problem.
    cin >> nums[0];
    cin >> nums[1];
    cin >> nums[2];

    // Checks if the sum of certainties is 2 or more, meaning at least two friends are sure.
    if (nums[0] + nums[1] + nums[2] >= 2) {
      counter++; // If so, increment the count of implemented problems.
    }
  }
  
  // Prints the total number of problems that will be implemented.
  cout << counter;
  
  return 0; // Indicates successful program execution.
}
