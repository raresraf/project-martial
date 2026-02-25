/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem from Codeforces (Problem 231A).
 *        This program determines the number of problems a team can implement.
 *        For each problem, three friends express their opinion (1 if sure, 0 if not).
 *        A problem is implemented if at least two friends are sure about the solution.
 * Algorithm: Simple Counting / Conditional Logic
 * Time Complexity: O(N), where N is the number of problems.
 * Space Complexity: O(1).
 */

#include <bits/stdc++.h> // Includes various standard libraries, e.g., iostream.
using namespace std;

/**
 * @brief Main function to read problem opinions and count solvable problems.
 * @return 0 if the program executes successfully.
 */
int main() {
  // Functional Utility: Disables synchronization between C-style I/O and C++ streams.
  // This, along with `cin.tie(nullptr)`, speeds up I/O operations,
  // which is crucial for competitive programming.
  ios::sync_with_stdio(false);
  // Functional Utility: Unties cin from cout.
  // This prevents `cout` from flushing before each `cin` operation, further
  // improving I/O performance.
  cin.tie(nullptr);

  // Declare variables:
  // n: total number of problems.
  // count: counter for the number of problems that can be implemented.
  // a1, a2, a3: opinions of the three friends for a single problem (1 for sure, 0 for not).
  int n, count = 0, a1, a2, a3;
  // Read the total number of problems from standard input.
  cin >> n;
  
  /**
   * Block Logic: Iterates through each problem to determine if it can be implemented.
   * Invariant: 'count' accumulates the number of problems that can be implemented.
   */
  for (int i = 0; i < n; ++i) { // Loop 'n' times, once for each problem.
    // Read the opinions of the three friends for the current problem.
    cin >> a1 >> a2 >> a3;
    
    // Block Logic: Check if at least two friends are sure about the solution.
    // Pre-condition: a1, a2, a3 are either 0 or 1.
    // Post-condition: 'count' is incremented if the sum of opinions is greater than 1.
    // A sum > 1 means at least two opinions are 1 (e.g., 1+1+0=2, 1+1+1=3).
    if (a1 + a2 + a3 > 1)
      count++; // If at least two friends agree, increment the solvable problems counter.
  }
  // Print the total number of problems that can be implemented, followed by a newline.
  cout << count << "\n";

  return 0; // Indicate successful program termination.
}

/*
231A-Team
time limit per test: 2 seconds
memory limit per test: 256 megabytes

Short Version:
Input: one line with n
then n lines with 3 numbers.
Return the amount of times more than one of these numbers is a 1.
*/