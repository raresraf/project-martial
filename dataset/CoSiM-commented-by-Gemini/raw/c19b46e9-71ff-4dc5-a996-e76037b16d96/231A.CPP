/**
 * @file 231A.CPP
 * @brief Solution for Codeforces Problem 231A, "Team".
 *
 * Algorithm:
 * This program calculates the number of problems a team of three will solve.
 * A team decides to solve a problem if at least two members are sure about the solution.
 * The input for each problem consists of three integers (0 or 1), where 1 means a member
 * is sure and 0 means they are not. The program iterates through all the problems,
 * sums the three opinions for each, and if the sum is 2 or greater, it increments a
 * counter for solved problems.
 *
 * Time Complexity: O(N), where N is the number of problems. The inner loops run a
 * constant number of times (3), so the overall complexity is linear with respect to N.
 *
 * Space Complexity: O(1). A vector of constant size (3) is used and cleared
 * within each iteration of the main loop, so space usage does not scale with N.
 */
//#include <bits/stdc++.h>
// using namespace std;
#include <iostream>
#include <vector>

int main() {
  /**
   * @brief Optimizes C++ stream I/O by decoupling from C standard streams.
   * This is a common practice in competitive programming for faster execution.
   */
  std::ios::sync_with_stdio(0);
  std.cin.tie(0);

  // n: total number of problems.
  // counter: counts the number of problems that will be implemented.
  // vector_sum: temporary sum of opinions for a single problem.
  int n, counter = 0, vector_sum = 0;
  // v: temporary vector to store the three opinions for one problem.
  std::vector<int> v;
  std::cin >> n;

  /**
   * @brief Main loop to iterate through each of the 'n' problems.
   */
  for (int j = 0; j < n; j++) {
    /**
     * @brief Reads the three opinions (0 or 1) for the current problem.
     */
    for (int i = 0; i < 3; i++) {
      int x;
      std::cin >> x;
      v.push_back(x);
    }

    /**
     * @brief Calculates the sum of the opinions stored in the vector.
     */
    for (int &it : v) {
      vector_sum += it;
    }
    
    // If two or more members are sure, the problem will be implemented.
    if (vector_sum >= 2) {
      counter++;
    }
    
    // Reset the sum and clear the vector for the next problem.
    vector_sum = 0;
    v.clear();
  }
  
  // Output the final count of problems to be implemented.
  std::cout << counter;

  return 0;
}
