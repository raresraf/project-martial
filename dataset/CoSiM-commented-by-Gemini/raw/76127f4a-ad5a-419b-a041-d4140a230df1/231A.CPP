/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem, counting problems solvable by a majority.
 *
 * This program reads an integer `n`, representing the number of problems. For each
 * problem, it reads the opinions of three team members (represented as 1 for 'sure'
 * and 0 for 'not sure'). It counts and prints the number of problems where at least
 * two members are sure.
 *
 * @algorithm
 * 1. Read `n`, the total number of problems.
 * 2. Initialize a counter `cont` to 0. This will track the number of problems
 *    the team will solve.
 * 3. Loop `n` times, processing one problem in each iteration.
 *    a. Read the three opinions (`i`, `j`, `k`) for the current problem.
 *    b. Sum the opinions.
 *    c. If the sum is greater than 1 (i.e., 2 or 3), it means at least two
 *       members are sure. Increment the counter `cont`.
 * 4. After the loop, print the final count `cont`.
 *
 * @complexity
 * Time Complexity: O(n), where `n` is the number of problems.
 * Space Complexity: O(1).
 */
#include <iostream>
using namespace std;

int main() {
  // i, j, k: store the opinion of each of the three members.
  // n: the number of problems.
  // cont: a counter for the problems that will be solved.
  int i, j, k, n, cont = 0;
  
  // Read the total number of problems.
  cin >> n;
  
  /**
   * @brief Main loop to iterate through each problem.
   */
  while (n--) {
    // Read the three members' opinions for the current problem.
    cin >> i >> j >> k;
    
    // Check if the sum of opinions is greater than 1 (i.e., 2 or 3).
    if (i + j + k > 1)
      // If at least two members are sure, increment the counter.
      cont++;
  }
  
  // Print the final count of problems to be solved.
  cout << cont << endl;
  
  return 0;
}
