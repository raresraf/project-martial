/**
 * @file 231A.CPP
 * @brief Counts the number of problems a team will solve based on member confidence.
 * 
 * This program solves a simple counting problem where a team of three decides
 * to implement a problem only if at least two of them are confident about the
 * solution.
 * 
 * @b Algorithm: The program iterates through each problem, summing up the
 *    confidence votes (1 for confident, 0 for not). If the sum is 2 or greater,
 *    it increments a counter for the total number of problems to be solved.
 * @b Time_Complexity: O(N), where N is the number of problems, as it requires a
 *    single pass through all the problems.
 * @b Space_Complexity: O(1), as it uses a constant amount of extra space regardless
 *    of the input size.
 */
#include <algorithm>
#include <ctype.h>
#include <iostream>
#include <map>
#include <stdio.h>
#include <string>

using namespace std;

int main() {
  int n; // Number of problems.
  cin >> n;
  int nr_problems = 0; // Counter for problems to be implemented.
  
  // Loop through each problem to check the team's confidence.
  for (auto i = 0; i < n; i++) {
    int aux1, aux2, aux3;
    // Read the confidence level of the three team members.
    cin >> aux1 >> aux2 >> aux3;
    
    // Check if at least two members are confident.
    if (aux1 + aux2 + aux3 >= 2) {
      nr_problems++;
    }
  }
  
  // Output the final count.
  cout << nr_problems;
  return 0;
}
