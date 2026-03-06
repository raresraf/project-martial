/**
 * @file 231A.CPP
 * @brief Counts the number of problems a team will solve based on member confidence.
 *
 * This program solves the "Team" problem from Codeforces. It reads a number of
 * sub-problems to consider. For each sub-problem, it reads the confidence (1 for
 * sure, 0 for not) of three team members. It then counts how many sub-problems
 * will be implemented, where a sub-problem is implemented if at least two members
 * are sure about the solution.
 *
 * @algorithm Iterative Counting
 * @complexity
 * Time: O(N), where N is the number of sub-problems, as each is processed once.
 * Space: O(1), as a fixed number of variables are used.
 */
#include <iostream>
using namespace std;

/**
 * @brief Main function to read problem data and calculate the result.
 */
int main() {
  // line_count: The total number of sub-problems to evaluate.
  int line_count;
  cin >> line_count;

  // member1, member2, member3: Store the confidence of each team member (1 or 0).
  // count: A counter for the number of problems the team will solve.
  int member1, member2, member3, count = 0;

  // Pre-condition: 'count' is 0.
  // Block Logic: Loop through each sub-problem.
  while (line_count--) {
    cin >> member1 >> member2 >> member3;
    // Check if the sum of confidence levels is 2 or greater.
    if (member1 + member2 + member3 >= 2)
      count++; // Increment the counter if at least two members are sure.
  }
  
  // Output the total count of problems that will be solved.
  cout << count;
  
  return 0;
}
