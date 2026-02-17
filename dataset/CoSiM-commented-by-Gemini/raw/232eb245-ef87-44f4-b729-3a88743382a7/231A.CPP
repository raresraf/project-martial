/**
 * @file 231A.CPP
 * @brief Solution for a programming problem that counts the number of teams confident enough to solve a problem.
 * @details This program reads an integer `n`, representing the number of problems. For each problem, it takes three integers (0 or 1) as input, representing the confidence of three team members. If at least two members are confident (input is 1), a counter is incremented. Finally, the total count of problems they are confident about is printed.
 *
 * @algorithm Iterative Counting
 * @time_complexity O(n) - The program iterates 'n' times, and for each iteration, it performs a constant number of operations (reading 3 integers and a check).
 * @space_complexity O(1) - The space used does not scale with the input size 'n', as variables are reused in each iteration.
 */
#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Processes 'n' teams' inputs to determine how many problems they will solve.
 * @param number The total number of problems/teams to process.
 * @return The number of problems that will be solved (where at least 2 members are sure).
 */
int TeamCpp(int number) {
  int res = 0; // Stores the count of problems to be solved.
  
  // Loop through each problem.
  for (int i = 0; i < number; i++) {
    int array[3];
    int count = 0; // Counter for confident members for the current problem.

    // Read the confidence level of each of the 3 members.
    for (int j = 0; j < 3; j++) {
      cin >> array[j];
      if (array[j] == 1) {
        count++;
      }
    }

    // If 2 or more members are confident, the team solves the problem.
    if (count >= 2) {
      res++;
    }
  }

  return res;
}

/**
 * @brief Main function to drive the program.
 * @return 0 on successful execution.
 */
int main() {
  int n; // Number of problems.
  cin >> n;
  
  // Calculate and store the result.
  int result = TeamCpp(n);
  
  // Output the final count.
  cout << result << endl;

  return 0;
}