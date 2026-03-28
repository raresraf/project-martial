/**
 * @file 231A.CPP
 * @brief A clean and efficient solution to the "Team" problem.
 *
 * This program determines how many problems a team of three will solve, based
 * on the rule that at least two members must be confident in the solution.
 *
 * @b Algorithm: The program iterates `n` times, once for each problem. In each
 *    iteration, it reads the three members' votes and checks if their sum is 2
 *    or greater. If it is, a counter is incremented.
 * @b Time_Complexity: O(N), where N is the number of problems, as it requires a
 *    single pass.
 * @b Space_Complexity: O(1), as it processes each problem's data without storing
 *    the entire set of votes.
 */
#include <iostream>
#include <vector>

using namespace std;

int main() {
  int n;
  int count = 0;
  cin >> n;
  
  // Loop `n` times to process each problem.
  while (n--) {
    int a, b, c;
    cin >> a >> b >> c;
    
    // The core logic: if the sum of votes is 2 or more, the team solves the problem.
    if (a + b + c >= 2) {
      count += 1;
    }
  }
  
  // Output the final number of problems to be solved.
  cout << count << endl;
}
