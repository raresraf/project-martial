/**
 * @file 231A.CPP
 * @brief Solution for the "Team" problem, with a setup for local testing.
 * @details This program counts how many problems a team of three will solve, where a
 * problem is solved if at least two members are confident (represented by 1). This version
 * includes a common competitive programming setup to redirect I/O to local files,
 * which is automatically disabled when compiled on an online judge platform.
 *
 * @section algorithm Algorithm
 * 1. Read the number of problems, `T`.
 * 2. Initialize a `count` of solvable problems to 0.
 * 3. Loop `T` times, once for each problem.
 *    a. Read the three integer decisions (0 or 1) for the current problem.
 *    b. Sum the three decisions.
 *    c. If the sum is greater than 1 (i.e., 2 or 3), it means at least two members
 *       agree, so the `count` is incremented.
 * 4. After the loop, the final `count` is printed.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(T), where T is the number of problems, as the program iterates through each problem once.
 * - Space Complexity: O(1), as the program only stores variables for the current problem being processed.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
/**
 * @brief Local I/O Redirection for Testing.
 * This preprocessor block is a common practice in competitive programming.
 * When the code is compiled locally (where ONLINE_JUDGE is not defined),
 * it redirects standard input from "input.txt" and standard output to "output.txt",
 * simplifying the testing process. This block has no effect when submitted to an online judge.
 */
#ifndef ONLINE_JUDGE
  // for getting input from input.txt
  freopen("input.txt", "r", stdin);
  // for writing output to output.txt
  freopen("output.txt", "w", stdout);
#endif

  int T;
  cin >> T;
  int count = 0;
  int a, b, c;

  // Loop through each problem case.
  for (int i = 0; i < T; i++) {

    cin >> a >> b >> c;

    // A problem is solved if the sum of confidences is greater than 1 (i.e., 2 or 3).
    if (a + b + c > 1) {
      count = count + 1;
    }
  }

  cout << count;

  return 0;
}