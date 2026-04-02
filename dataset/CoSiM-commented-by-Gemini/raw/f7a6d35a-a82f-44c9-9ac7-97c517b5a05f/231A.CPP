/**
 * @file 231A.CPP
 * @brief Solution to a competitive programming problem.
 * 
 * Problem Statement:
 * A team of three friends is participating in a programming contest. They decide to solve
 * a problem if at least two of them are sure about the solution. For each problem,
 * we are given three boolean values (0 or 1) representing whether each friend is sure.
 * The task is to count the number of problems the team will solve.
 *
 * Algorithm:
 * The program reads the total number of problems 'n'. It then iterates 'n' times.
 * In each iteration, it reads the three boolean values for a problem. It checks if the
 * sum of these three values is 2 or more. If it is, a counter is incremented.
 * Finally, the total count is printed.
 *
 * Time Complexity: O(n), as the program iterates through each of the 'n' problems once.
 * Space Complexity: O(1), as the memory usage is constant and does not depend on 'n'.
 */
#include <iostream>

using namespace std;

int main() {
  // 'n' is the number of problems in the contest.
  int n;
  cin >> n;
  
  // 'count' will store the number of problems to be implemented.
  int count = 0;
  
  /**
   * Block Logic: Iterate through each problem to decide if it will be solved.
   * Pre-condition: 'n' contains the total number of problems.
   * Invariant: 'count' holds the number of problems solved so far.
   */
  while (n--) {
    // 'a', 'b', 'c' represent the confidence of the three friends.
    int a, b, c;
    cin >> a >> b >> c;
    
    // A problem is solved if at least two friends are sure.
    // The sum will be >= 2 if two or three friends are sure (value is 1).
    if ((a + b + c) > 1)
      count++;
  }
  
  // Output the total number of problems the team will solve.
  cout << count;
  return 0;
}
