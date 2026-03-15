/**
 * @file 231A.CPP
 * @brief A concise solution to the "Team" problem.
 * @details The program determines how many problems a three-person team will solve.
 * A problem is solved if at least two members are sure about the solution.
 *
 * @algorithm
 * This solution uses a direct and efficient approach:
 * 1. Read the total number of problems, `n`.
 * 2. Loop `n` times, once for each problem.
 * 3. In each iteration, read the three members' opinions (0 or 1) into variables `p`, `v`, and `t`.
 * 4. Sum the three opinions and check if the result is 2 or greater.
 * 5. If the condition is true, increment a `sum` counter.
 * 6. After the loop, print the final `sum`.
 *
 * @complexity
 * - Time: O(N), where N is the number of problems, as each problem is processed in constant time.
 * - Space: O(1), as a fixed number of variables are used regardless of input size.
 */
#include <iostream>
using namespace std;

int main() {
  // n: The total number of problems to be evaluated.
  // p, v, t: Variables to store the opinion of Petya, Vasya, and Tonya (1 for sure, 0 for not).
  // i: Loop counter.
  // sum: A counter for the number of problems that will be implemented.
  int n, p, v, t, i, sum = 0;
  cin >> n;

  /**
   * @brief This loop processes each of the `n` problems.
   */
  for (i = 0; i < n; i++) {
    // Read the three opinions for the current problem.
    cin >> p >> v >> t;
    
    // Pre-condition: Check if the sum of opinions is at least 2.
    // This is the most direct way to check if two or more members are sure.
    if (p + v + t >= 2) {
      // Post-condition: If they are, increment the count of problems to be solved.
      sum = sum + 1;
    }
  }

  // Output the final count.
  cout << sum;
  return 0;
}