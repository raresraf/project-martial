/**
 * @file 231A.CPP
 * @brief Solution to Codeforces Problem 231A: "Team".
 *
 * This problem involves determining how many problems a team of three friends
 * will implement. The team implements a problem if at least two of them are
 * sure about the solution.
 *
 * Algorithm: Counting sure friends per problem.
 * For each problem, the program reads three integers representing the friends'
 * certainty (1 if sure, 0 if not sure). It then sums these values. If the sum
 * is 2 or more, the problem is implemented, and a counter is incremented.
 *
 * Time Complexity: O(T), where T is the number of test cases (problems), as
 *                  each test case involves a constant number of arithmetic
 *                  operations and comparisons.
 * Space Complexity: O(1), as only a few variables are used.
 */
#include <bits/stdc++.h> // Includes various standard libraries, such as iostream for input/output.
using namespace std;     // Uses the standard namespace to avoid prefixing std::

/**
 * @brief Main function to solve the "Team" problem.
 * Reads the number of problems, and for each problem, determines if the team
 * will implement it based on the certainty of three friends.
 *
 * @return int Returns 0 on successful execution.
 */
int main() {
  int t; // Stores the number of test cases (problems).
  int a, b, c; // Stores the certainty of each of the three friends for a problem (1 for sure, 0 for not).
  int ck = 0; // Counter for the number of problems the team will implement.
  scanf("%d", &t); // Reads the total number of problems.
  
  /**
   * Block Logic: Processes each problem (test case) one by one.
   * Pre-condition: 't' specifies the number of problems to evaluate.
   * Invariant: After each iteration, one problem has been assessed.
   */
  while (t--) {
    cin >> a >> b >> c; // Reads the certainty of the three friends for the current problem.
    // Block Logic: Checks if at least two friends are sure about the solution.
    if ((a + b + c) >= 2) {
      ck++; // Increments the counter if the problem will be implemented.
    }
  }
  cout << ck; // Prints the total number of problems the team will implement.
  return 0; // Indicates successful program execution.
}
