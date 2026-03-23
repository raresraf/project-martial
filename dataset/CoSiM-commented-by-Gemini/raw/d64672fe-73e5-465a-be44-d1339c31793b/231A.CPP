/**
 * @file 231A.CPP
 * @brief Solution for Codeforces Problem 231A, "Team".
 * @details This program counts how many problems a team will solve based on the
 * confidence of its three members. A problem is solved if at least two members
 * are confident. This implementation reads the three confidence values for each
 * problem into a small array.
 *
 * @algorithm Simple Counting
 * @complexity
 * Time: O(N), where N is the number of problems, due to the single loop.
 * Space: O(1), as the small array is reused in each iteration.
 */
#include <iostream>

using namespace std;

int main() {
  int t;         // 't' for the number of test cases (problems).
  int array[3];  // An array to hold the three members' opinions for one problem.
  int num = 0;   // A counter for the number of solvable problems.
  
  cin >> t;

  /**
   * @brief A loop that runs 't' times to process each problem.
   * The loop continues as long as 't' is greater than 0, decrementing 't' each time.
   */
  for (; t > 0; t--) {
    // Read the three opinions into the array.
    cin >> array[0] >> array[1] >> array[2];

    // Check if the sum of opinions is 2 or more.
    if (array[0] + array[1] + array[2] >= 2) {
      num++; // Increment the counter for solvable problems.
    }
  }

  // Print the final count.
  cout << num << endl;
  
  return 0; // Indicate successful execution.
}