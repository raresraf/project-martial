/**
 * @file 231A.CPP
 * @brief Solution for Codeforces Problem 231A - "Team".
 * @details The problem asks to count the number of problems that a three-person team will implement.
 * The team decides to implement a problem if at least two of the three members are sure about the solution.
 * The input consists of n problems, and for each problem, three boolean values (0 or 1) indicating
 * if each member is sure.
 *
 * @author Sourabh Bollapragada
 *
 * @algorithm
 * 1. Read the total number of problems, `n`.
 * 2. Initialize a counter `c` to 0, which will store the number of problems to be implemented.
 * 3. Loop `n` times, once for each problem.
 * 4. In each iteration, read the three members' opinions (`p`, `v`, `t`).
 * 5. Sum the three opinions. If the sum is 2 or greater, it means at least two members are sure.
 * 6. If the condition is met, increment the counter `c`.
 * 7. After the loop, print the final count `c`.
 *
 * @complexity
 * Time Complexity: O(n), where n is the number of problems, because the program iterates through each problem once.
 * Space Complexity: O(1), as it uses a fixed amount of memory regardless of the input size.
 */
//
// Created by Sourabh Bollapragada on 6/3/20.
//

// Problem Link https://codeforces.com/problemset/problem/231/A

#include <iostream>

int main() {
  int n;
  int p, v, t;
  std::cin >> n;
  int c = 0; // Counter for problems that will be implemented.
  /**
   * Block Logic: This loop processes each of the 'n' problems.
   * Pre-condition: 'n' is the total number of problems.
   * Invariant: 'c' holds the count of problems to be implemented from the first 'i' problems.
   */
  for (int i = 0; i < n; i++) {
    // Read the opinions of the three team members for the current problem.
    std::cin >> p >> v >> t;
    /**
     * Block Logic: Checks if the team will implement the problem.
     * The condition `p + v + t > 1` is equivalent to checking if at least two members
     * are sure about the solution (i.e., their input is 1).
     */
    if (p + v + t > 1) {
      c++;
    }
  }
  // Print the total number of problems to be implemented.
  std::cout << c << std::endl;
}
