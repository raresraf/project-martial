/**
 * @file 231A.CPP
 * @brief Counts the number of problems that will be implemented based on team confidence.
 *
 * This program determines how many problems, out of a given set, a team of three
 * will decide to implement. The condition for implementation is that at least two
 * out of the three members are confident about the solution. Each problem's confidence
 * level is represented by three boolean-like integers (0 or 1).
 *
 * @algorithm This is a direct and efficient implementation. It iterates through each
 * problem, reads the three confidence values into an array, and sums them up.
 * If the sum is 2 or greater, it increments a counter. The program also includes
 * a check to ensure the number of problems `n` is within the valid problem constraints.
 *
 * @time_complexity O(N), where N is the number of problems, as it involves a single
 * loop through all the problems.
 * @space_complexity O(1), as it uses a small, fixed-size array `arr` that is
 * reused for each problem.
 */
#include <iostream>
using namespace std;
int main() {
  int arr[3];    // Array to store the confidence levels of the three members.
  int i, j;      // Loop counters.
  int n;         // The total number of problems to evaluate.
  int flag = 0;  // Counter for the number of problems to be implemented.
  
  cin >> n;

  // Check if the number of problems is within the specified constraints [1, 1000].
  if (1 <= n && n <= 1000) {
    /**
     * @brief Main loop to process each problem.
     * @invariant `flag` holds the count of implementable problems for the first `i` problems.
     */
    for (i = 0; i < n; i++) {
      // Read the three confidence values for the current problem.
      for (j = 0; j < 3; j++) {
        cin >> arr[j];
      }
      
      /**
       * @brief Check if at least two members are confident by summing their votes.
       * @pre arr contains three integer values (0 or 1).
       */
      if (arr[0] + arr[1] + arr[2] >= 2)
        flag++; // Increment the counter if the condition is met.
    }
    cout << flag << endl;
  }
  return 0;
}
