/**
 * @file 231A.CPP
 * @brief Solution for a problem that involves counting based on a sum condition.
 * @details This program reads a number of problems, `n`. For each problem, it reads the opinions
 * of three judges (represented by 1 for "sure" and 0 for "not sure"). If at least two out of the three
 * judges are sure about the solution, the team decides to solve the problem. The program counts
 * how many problems the team will solve.
 *
 * @section algorithm Algorithm
 * 1. Read the total number of problems, `n`.
 * 2. Initialize a counter, `flag` (or `solved_problems`), to 0.
 * 3. Loop `n` times, once for each problem:
 *    a. Read the three opinions (0 or 1) for the current problem.
 *    b. Calculate the sum of the three opinions.
 *    c. If the sum is 2 or greater, it means at least two judges are sure. In this case, increment the counter.
 * 4. After the loop, print the final count of problems to be solved.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(n), where `n` is the number of problems. The program iterates through each problem exactly once.
 * - Space Complexity: O(1), as the memory usage is constant and does not depend on the input size.
 */
#include <iostream>

using namespace std;

/**
 * @brief Main function to execute the problem-solving logic.
 * @return 0 on successful execution.
 */
int main() {
  int n;
  cin >> n; // Number of problems

  int solved_problems = 0; // Counter for problems they are going to solve

  // Loop through each problem.
  for (int i = 0; i < n; i++) {
    int a, b, c;
    cin >> a >> b >> c; // Read the opinions of the three judges.
    // If the sum of opinions is 2 or more, they solve the problem.
    if (a + b + c >= 2) {
      solved_problems++;
    }
  }
  
  // Print the total number of problems to be solved.
  cout << solved_problems;
  return 0;
}
