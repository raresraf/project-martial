/**
 * @file 231A.CPP
 * @brief Solution for the "Team" problem (Codeforces 231A).
 * @author competitive-programmer
 *
 * @details This program calculates the number of problems a three-person team
 * will implement. The team decides to solve a problem only if at least two of
 * its members are sure about the solution (represented by a '1').
 *
 * Algorithm (Direct Summation):
 * 1. Read the total number of problems (`testcases`).
 * 2. Initialize a counter for the final answer, `solutions`, to zero.
 * 3. Loop for each problem.
 * 4. In each iteration, read the three opinions (0 or 1) of the team members.
 * 5. Sum the three opinions. Since the values are 0 or 1, the sum directly
 *    equals the number of members who are sure.
 * 6. If the sum is 2 or greater, it means at least two members are sure, so
 *    increment the `solutions` counter.
 * 7. After the loop, print the final value of `solutions`.
 *
 * Time Complexity: O(n), where n is the number of problems (`testcases`).
 * Space Complexity: O(1).
 */
#include <iostream>
using namespace std;
int main() {
  // `testcases` is the number of problems to evaluate.
  int testcases;
  cin >> testcases;
  
  // `Petya`, `Vasya`, `Tonya` hold the opinions (0 or 1) for each problem.
  // `solutions` counts the number of problems to be implemented.
  int Petya, Vasya, Tonya, solutions = 0;
  
  /**
   * Block Logic: This loop iterates once for each problem.
   */
  while (testcases > 0) {
    cin >> Petya >> Vasya >> Tonya;
    
    /**
     * Decision Logic: Summing the three opinions (which are 0 or 1) directly
     * gives the count of members who are sure. If this count is 2 or more,
     * the team will solve the problem.
     */
    if ((Petya + Vasya + Tonya) >= 2) {
      solutions++;
    }
    testcases--;
  }
  
  // Print the final count.
  cout << solutions;
  return 0;
}
