/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem from competitive programming platforms.
 * 
 * The problem is to count how many problems a team of three will decide to solve.
 * The rule is that they solve a problem if at least two members are sure about
 * the solution.
 *
 * @b Algorithm: The program iterates through each problem, summing up the
 *    confidence votes (1 for confident, 0 for not) of the three members.
 *    If the sum is 2 or greater, a counter is incremented.
 * @b Time_Complexity: O(N), where N is the number of problems, due to the
 *    single loop.
 * @b Space_Complexity: O(1), as it uses a constant amount of memory.
 */
// 4006945   Jul 3, 2013 6:18:52 PM	fuwutu	 231A - Team	 GNU C++0x
// Accepted	15 ms	0 KB
#include <iostream>

using namespace std;

int main() {
  int n; // Number of problems.
  int Petya, Vasya, Tonya; // Votes of the three team members.
  int number(0); // Counter for problems to be implemented.
  
  cin >> n;
  
  // Loop n times, once for each problem.
  while (n--) {
    cin >> Petya >> Vasya >> Tonya;
    // Check if the sum of votes is at least 2.
    if (Petya + Vasya + Tonya >= 2) {
      number += 1;
    }
  }
  
  // Output the final count of problems to be solved.
  cout << number << endl;
  return 0;
}