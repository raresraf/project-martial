/**
 * @file 231A.CPP
 * @brief Solution to the "Team" competitive programming problem.
 * @details This program calculates the number of problems a team of three will
 * implement, based on the condition that at least two members must be sure
 * about the solution.
 *
 * @section algorithm Algorithm
 * A simple counting approach is used:
 * 1. Read the total number of problems, `n`.
 * 2. Initialize a counter, `solved_problems_count`, to 0. This will track the
 *    number of problems the team will solve.
 * 3. Loop `n` times, once for each problem.
 * 4. In each iteration, read the three members' opinions (1 for sure, 0 for not).
 * 5. Check if the sum of the three opinions is greater than 1 (i.e., 2 or 3),
 *    which means at least two members agree.
 * 6. If the condition is met, increment the `solved_problems_count` counter.
 * 7. After the loop, print the final count.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(N), where N is the number of problems, due to the single loop.
 * - Space Complexity: O(1), as memory usage is constant.
 */
#include <iostream>
using namespace std;

int main() {
  int num_problems; // The total number of problems.
  cin >> num_problems;

  int solved_problems_count = 0; // Counter for problems to be implemented.
  
  // Loop through each problem to read the team's opinions.
  for (int i = 0; i < num_problems; i++) {
    int petya, vasya, tonya; // Variables for the three members' opinions.
    cin >> petya >> vasya >> tonya;
    
    // Check if the sum of opinions is greater than 1 (i.e., at least 2).
    if ((petya + vasya + tonya) > 1) {
      solved_problems_count++;
    }
  }
  
  // Print the final count of problems the team will solve.
  cout << solved_problems_count;
}