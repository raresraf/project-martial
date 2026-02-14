/**
 * @file 231A.CPP
 * @brief Solution for the "Team" problem, which counts problems to be solved based on confidence.
 * @details This program determines how many problems a team will solve. A problem is solved
 * if at least two out of three team members are confident in the solution. This implementation
 * uses an unconventional loop structure to process the inputs.
 *
 * @section algorithm Algorithm
 * The logic iterates through all individual inputs (3 for each problem) in a single flat loop.
 * 1. Read the number of problems, `n`.
 * 2. Initialize a counter for solved problems, `ans`, to 0.
 * 3. Initialize a temporary counter, `ni`, to 0. This tracks the number of confident members for the current problem.
 * 4. A single `for` loop runs `n * 3` times to read all `n*3` integer inputs.
 * 5. Inside the loop, it uses the modulo operator (`i % 3`) to identify the end of each problem's inputs.
 *    - For the first and second inputs of a problem (`i % 3 != 0`), it reads the value and increments `ni` if it's 1.
 *    - For the third input (`i % 3 == 0`), it reads the value, increments `ni` if it's 1, and then immediately checks if the total `ni` for that problem is 2 or more.
 *    - If the confidence level is sufficient, the main answer `ans` is incremented.
 *    - After checking, `ni` is reset to 0 to prepare for the next problem's inputs.
 * 6. After the loop, the final count `ans` is printed.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(n), as the loop runs `3*n` times, which is linearly proportional to `n`.
 * - Space Complexity: O(1), as it uses a fixed number of variables.
 */
#include <iostream>
using namespace std;

/**
 * @brief Main function to count the number of solvable problems.
 * @return 0 on successful execution.
 */
int main() {
  int n = 0;      // Number of problems.
  int ni = 0;     // Counter for confident members for the current problem.
  int ans = 0;    // Total count of problems to be solved.
  int i = 0;      // Loop counter.
  int x = 0;      // Variable to store each input opinion.
  
  cin >> n;
  
  // This loop iterates 3*n times, processing one opinion at a time.
  for (i = 1; i <= n * 3; i++) {
    cin >> x;
    if (x == 1) {
      ni++;
    }
    
    // Check if it's the third opinion for a problem.
    if (i % 3 == 0) {
      // If at least two members are confident, increment the answer.
      if (ni >= 2) {
        ans++;
      }
      // Reset the confidence counter for the next problem.
      ni = 0;
    }
  }
  
  cout << ans << endl;
  return 0;
}
