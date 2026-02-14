/**
 * @file 231A.CPP
 * @brief Solution for the "Team" problem, counting problems to be solved.
 * @details This program determines how many problems a team of three will solve.
 * The team decides to solve a problem only if at least two of the three members are confident
 * about the solution. The confidence of each member is given as a binary input (1 for confident, 0 for not).
 *
 * @section algorithm Algorithm
 * 1. The program reads an integer `n`, the total number of problems.
 * 2. It initializes a counter `cnt` to zero, which will store the number of problems to be solved.
 * 3. It then enters a loop that runs `n` times, once for each problem.
 * 4. In each iteration, it reads the confidence levels of the three members (`p`, `v`, `t`).
 * 5. It checks if the sum of their confidence levels (`p + v + t`) is greater than or equal to 2.
 *    This is a concise way to determine if at least two members are confident.
 * 6. If the condition is met, the counter `cnt` is incremented.
 * 7. After the loop finishes, the total count `cnt` is printed.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(n), where `n` is the number of problems, because the program processes each problem once.
 * - Space Complexity: O(1), as it uses a fixed amount of memory regardless of the input size.
 */
#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Main function to execute the logic for counting solvable problems.
 * @return 0 on successful execution.
 */
int main() {
  int n, p, v, t, cnt;
  
  // This outer loop is a common pattern for handling multiple test cases,
  // though for this problem, typically only one value of 'n' is processed.
  while (cin >> n) {
    cnt = 0; // Initialize a counter for solvable problems.
    
    // Loop through each of the 'n' problems.
    while (n--) {
      cin >> p >> v >> t; // Read confidence levels of the three members.
      
      // Check if the sum of confidence levels is at least 2.
      if (p + v + t >= 2) {
        cnt++; // If so, increment the counter.
      }
    }
    // Print the total number of problems the team will solve.
    cout << cnt << endl;
  }
  return 0;
}
