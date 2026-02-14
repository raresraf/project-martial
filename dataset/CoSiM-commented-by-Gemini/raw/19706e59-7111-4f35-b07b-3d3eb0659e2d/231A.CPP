/**
 * @file 231A.CPP
 * @brief Solution for the "Team" problem, which counts problems to be implemented.
 * @details The program determines how many problems a team of three will solve.
 * A problem is solved if at least two of the three members are confident about the
 * solution (represented by 1).
 *
 * @section algorithm Algorithm
 * 1. An integer `a` is read, representing the total number of problems.
 * 2. A global variable `sum` is initialized to 0 to count the number of solvable problems.
 * 3. The program loops `a` times, once for each problem.
 * 4. In each iteration, it reads the three integer opinions (`b`, `c`, `d`) for the current problem.
 * 5. It checks if the sum of the opinions (`b + c + d`) is greater than 1 (i.e., 2 or 3).
 * 6. If the condition is true, it means at least two members are confident, and the `sum` is incremented.
 * 7. After the loop completes, the final `sum` is printed.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(a), where `a` is the number of problems, as the program iterates through each problem once.
 * - Space Complexity: O(1), as it uses a fixed number of global variables.
 */
#include <iostream>
using namespace std;

// Global variables for problem count, opinions, and the final answer.
int a, b, c, d, sum = 0;

/**
 * @brief Main function to execute the problem-solving logic.
 * @return 0 on successful execution.
 */
int main() {
  // Read the total number of problems.
  cin >> a;
  
  // Loop through each problem.
  for (int i = 0; i < a; i++) {
    // Read the three opinions for the current problem.
    cin >> b >> c >> d;
    
    // Check if at least two members are confident.
    if ((b + c + d) > 1) {
      sum++;
    }
  }
  
  // Print the total number of problems to be solved.
  cout << sum;
  
  return 0; // Though not strictly necessary as main implicitly returns 0.
}