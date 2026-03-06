/**
 * @file 231A.CPP
 * @brief Counts the number of problems a team will solve based on member confidence.
 *
 * This program solves the "Team" problem from Codeforces. It determines how many
 * problems a team of three will solve, given that they only proceed with a
 * problem if at least two members are confident in the solution.
 *
 * @algorithm Iterative Counting
 * @complexity
 * Time: O(N), where N is the number of problems, as each is processed once.
 * Space: O(1), using a fixed number of variables.
 */
#include <iostream>
using namespace std;

/**
 * @brief Main function to read problem data and calculate the result.
 */
int main() {
  // n: the total number of problems.
  // a[5]: an unused array, shadowed by local variables inside the loop.
  // solved: a counter for the problems the team will implement.
  int n, a[5], solved = 0;
  cin >> n;
  
  // Pre-condition: 'solved' is 0.
  // Block Logic: Loop 'n' times, once for each problem.
  while (n--) {
    // a, b, c: store the confidence of each team member (1 for sure, 0 otherwise).
    int a, b, c;
    cin >> a >> b >> c;
    
    // The commented-out code below represents a more verbose but logically
    // equivalent way to check if at least two members agree.
    // if(((a[0]==1) && (a[1]==1)) || ((a[1]==1) && (a[2]==1)) || ((a[2]==1) && (a[0]==1)) )
    
    // A concise check: if the sum of confidence levels is 2 or more,
    // the team solves the problem.
    if (a + b + c >= 2)
      solved++;
  }
  
  // Output the total count of solved problems.
  cout << solved << endl;
  
  return 0;
}
