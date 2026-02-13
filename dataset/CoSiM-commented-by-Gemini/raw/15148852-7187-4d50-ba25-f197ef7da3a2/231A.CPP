/**
 * @file 231A.CPP
 * @brief A space-efficient solution for the "Team" problem.
 * @details This program counts the number of problems a team of three will solve.
 * The team decides to implement a problem if at least two of its three members are
 * confident about the solution. This implementation is space-efficient as it
 * processes each problem's data on the fly without storing the entire input set.
 *
 * @section algorithm Algorithm
 * 1. Read the total number of problems, `n`.
 * 2. Initialize a `sum` counter to 0, which will track the number of solvable problems.
 * 3. Loop `n` times, once for each problem:
 *    a. Read the three integer decisions (0 or 1) for the current problem.
 *    b. The core logic `a + b + c >= 2 ? sum++ : sum;` uses a ternary operator.
 *       It checks if the sum of the three decisions is 2 or greater.
 *    c. If the condition is true, it increments `sum`. The `else` part (`: sum`) does nothing.
 * 4. After processing all `n` problems, the final `sum` is printed.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(N), where N is the number of problems, because it iterates through the problems once.
 * - Space Complexity: O(1), because it only stores the decisions for one problem at a time, making it highly memory-efficient.
 */
#include <iostream>
using namespace std;
int main() {

  int n;
  int sum = 0; // Counter for the number of problems the team will solve.
  cin >> n;
  // Loop through each of the 'n' problems.
  for (int i = 0; i < n; i++) {
    int a, b, c;
    // Read the decisions of the three members for the current problem.
    cin >> a >> b >> c;
    // Use a ternary operator to increment the sum if at least two members are sure.
    // The condition `a + b + c >= 2` evaluates to true if two or three members agree.
    a + b + c >= 2 ? sum++ : sum;
  }

  // Print the total count of solvable problems.
  cout << sum;

  return 0;
}
