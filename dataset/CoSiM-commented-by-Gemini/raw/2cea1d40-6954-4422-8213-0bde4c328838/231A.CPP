/**
 * @file 231A.CPP
 * @brief Solution to a programming problem about team decision making.
 * @details This program counts the number of problems a team of three will decide to implement.
 * The decision to implement a problem is made if at least two out of the three members are confident about its solution.
 *
 * Algorithm:
 * 1. Read the total number of problems, `n`.
 * 2. Initialize a counter `answ` to 0, which will store the final count of problems to be implemented.
 * 3. Loop `n` times to process each problem.
 * 4. For each problem, read the three members' opinions, represented by integers (1 for confident, 0 for not confident).
 * 5. Count how many members are confident (i.e., how many of the inputs are 1).
 * 6. If this count is 2 or greater, it means the team will implement the solution. Increment the `answ` counter.
 * 7. After processing all `n` problems, print the final value of `answ`.
 *
 * Time Complexity: O(n), as the program iterates through the `n` problems, and the inner loop runs a fixed 3 times.
 * Space Complexity: O(n), due to the `data[1000][5]` array used to store the inputs. This could be optimized to O(1) by processing each line's input without storing it.
 */
#include <iostream>
using namespace std;
int main() {
  int n, count = 0, answ = 0, data[1000][5];
  cin >> n;

  /**
   * Block Logic: Iterate through each problem to determine if it will be implemented.
   * Pre-condition: 'n' is the number of problems.
   * Invariant: 'answ' holds the count of problems that will be implemented from the ones processed so far.
   */
  for (int i = 0; i < n; i++) {
    count = 0; // Reset confidence counter for each new problem.
    
    // Read the opinions of the three members for the current problem.
    for (int j = 0; j < 3; j++) {
      cin >> data[i][j];
      if (data[i][j] == 1)
        count++; // Increment if a member is confident.
    }
    
    // If two or more members are confident, the team implements the problem.
    if (count >= 2)
      answ++;
  }
  
  // Print the total number of problems to be implemented.
  cout << answ;
  return 0;
}
