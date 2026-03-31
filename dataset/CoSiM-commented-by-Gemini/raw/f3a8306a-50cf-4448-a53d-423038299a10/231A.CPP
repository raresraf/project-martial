/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem by counting confident members for each sub-problem.
 *
 * This program determines the number of problems a three-person team will solve.
 * It reads an integer `n` (number of problems) and then loops `n` times. In each
 * iteration, it reads the three members' confidence levels (1 for sure, 0 for not).
 * If two or more members are sure, a counter for solved problems is incremented.
 * The final count is printed.
 *
 * @author f3a8306a-50cf-4448-a53d-423038299a10
 *
 * @algorithm Nested Loop Counting
 * @complexity
 * Time: O(N), for iterating through N problems, with a constant-time inner loop.
 * Space: O(1), as memory usage is constant.
 */
#include <iostream>
using namespace std;

/**
 * @note The use of `main()` without a return type is non-standard in C++.
 * Modern C++ requires `int main()`. Compilers often default it to `int`.
 */
main() {
  // `n`: number of problems. `i`: loop counter. `total`: solved problems counter.
  int n, i, total = 0;
  cin >> n;
  /**
   * @brief Main loop to process each of the `n` problems.
   */
  while (n > 0) {
    int arr[3]; // Array to store the three opinions for one problem.
    int x = 0;  // Counter for confident members for the current problem.
    
    /**
     * @brief Inner loop to read the three opinions and count confident members.
     */
    for (i = 0; i < 3; i++) {
      cin >> arr[i];
      if (arr[i] == 1)
        x = x + 1;
    }
    
    /**
     * @brief Check if at least two members are confident.
     */
    if (x >= 2) {
      total = total + 1;
    }
    
    // Manually decrement the loop counter.
    n = n - 1;
  }
  // Print the total number of problems the team will solve.
  cout << total << endl;
}