/**
 * @file 231A.CPP
 * @brief A structured solution to the "Team" problem (Codeforces 231A).
 *
 * @algorithm
 * The problem is to count the number of problems a three-person team will solve,
 * where a problem is solved if at least two members are confident.
 *
 * This implementation uses a modular approach with a helper function:
 * 1. Read the total number of problems, `n`.
 * 2. Store all `n` sets of three opinions in a 2D array `data[n][3]`. Note that
 *    this uses a Variable Length Array (VLA), a non-standard C++ extension.
 * 3. Iterate through each set of opinions in the 2D array.
 * 4. For each set, call a helper function `can_be_implemented`.
 * 5. The helper function counts the number of '1's (confident members) and
 *    returns `true` if the count is 2 or more.
 * 6. If the helper function returns `true`, increment a solution counter.
 * 7. After the loop, print the final counter.
 *
 * @complexity
 * - Time Complexity: O(N), for iterating through the N problems.
 * - Space Complexity: O(N), due to storing all `n` sets of inputs in a 2D array.
 */
#include <iostream>

using namespace std;
// Function prototype
bool can_be_implemented(int *);

int main(void) {
  int n;
  cin >> n;

  // Define a 2D Variable Length Array (VLA) to store all inputs.
  // Note: VLAs are a non-standard extension in C++.
  int data[n][3];

  /**
   * @brief Read all n*3 inputs into the 2D array.
   */
  for (int i = 0; i < n; i++)
    cin >> data[i][0] >> data[i][1] >> data[i][2];

  // process data
  int implement_solution_count = 0;

  /**
   * @brief Process each problem by calling the helper function.
   */
  for (int i = 0; i < n; i++) {
    // data[i] decays to a pointer to the first element of the i-th row.
    if (can_be_implemented(data[i]))
      implement_solution_count++;
  }

  cout << implement_solution_count << endl;

  return 0;
}

/**
 * @brief Determines if a problem can be implemented based on team confidence.
 * @param arr A pointer to an integer array containing 3 elements (the opinions).
 * @return true if two or more members are confident (value is 1).
 * @return false otherwise.
 */
bool can_be_implemented(int *arr) {
  int one_count = 0;
  // Count how many members are confident.
  for (int i = 0; i < 3; i++)
    if (arr[i] == 1)
      one_count++;

  if (one_count >= 2)
    return true;
  else
    return false;
}