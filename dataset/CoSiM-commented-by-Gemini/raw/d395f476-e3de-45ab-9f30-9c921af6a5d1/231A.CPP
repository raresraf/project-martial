/**
 * @file 231A.CPP
 * @brief A straightforward solution to the "Team" problem from Codeforces (Problem 231A).
 * @details The program counts the number of problems a team will solve. The condition
 * for solving a problem is that at least two of the three members are confident
 * in the solution.
 *
 * @section algorithm
 * Algorithm: Iterative Processing
 * 1. The total number of problems, N, is read.
 * 2. The program loops N times, processing one problem in each iteration.
 * 3. Inside the loop, it reads the three members' opinions (0 or 1).
 * 4. It counts how many of these opinions are '1' (confident).
 * 5. If this count is 2 or greater, a total counter for solvable problems is
 *    incremented.
 * 6. After the loop, the total count is printed.
 *
 * @section complexity
 * - Time Complexity: O(N), as each of the N problems is processed in a fixed
 *   number of operations (3 reads, a comparison, and an increment).
 * - Space Complexity: O(1), as the program uses a constant amount of memory
 *   regardless of the number of problems.
 */
#include <iostream>
using namespace std;
int main() {
  int n;
  int arr[3];
  cin >> n;
  int count = 0;
  int total = 0;
  /**
   * @brief This loop processes each of the N problems.
   * @invariant `total` stores the number of problems solvable so far.
   */
  while (n--) {
    /**
     * @brief Reads the three opinions for the current problem and counts the 'sure' ones.
     * The `count` variable is reset for each new problem.
     */
    for (int i = 0; i < 3; i++) {
      cin >> arr[i];
      if (arr[i] == 1) {
        count++;
      }
    }
    
    /**
     * @brief Checks if at least two members are sure about the current problem's solution.
     * If the condition is met, the total number of solvable problems is incremented.
     */
    if (count >= 2) {
      total++;
    }
    
    // Reset the counter for the next problem.
    count = 0;
  }
  
  cout << total << endl;

  return 0;
}