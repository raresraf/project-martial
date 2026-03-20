/**
 * @file 231A.CPP
 * @brief Solution for the "Team" problem (Codeforces 231A).
 * @details This program counts how many problems a team of three will implement.
 * The condition for implementation is that at least two of the three members
 * are confident in the solution (represented by a '1' input).
 *
 * @section algorithm Algorithm
 * 1. Read the total number of problems, 'n'.
 * 2. Initialize a counter to zero.
 * 3. Loop 'n' times for each problem:
 *    a. Read the three members' inputs (0 or 1).
 *    b. Sum the three inputs. The sum represents the number of confident members.
 *    c. If the sum is greater than 1 (i.e., 2 or 3), increment the counter.
 * 4. Print the final count.
 * This approach is very direct and efficient.
 *
 * @section complexity Complexity
 * - Time Complexity: O(N), where N is the number of problems.
 * - Space Complexity: O(1).
 */
#include <iostream>

using namespace std;

/**
 * @brief Handles the core logic for a single test case.
 */
void solve() {
  int n; // Number of problems to evaluate.
  cin >> n;
  int q = 0; // Counter for the number of problems to be implemented.

  // Loop through each problem.
  for (int i = 0; i < n; i++) {
    int a, b, c; // Confidence of the three team members.
    cin >> a >> b >> c;
    // Sum the inputs. This gives the number of members who are sure.
    int s = a + b + c;
    // If two or more members are sure, the team will implement the problem.
    if (s > 1) {
      ++q;
    }
  }
  // Print the final count.
  cout << q;
}

/**
 * @brief Main function, entry point of the program.
 */
int main() {
  solve();
  return 0;
}
