/**
 * @file 231A.CPP
 * @brief Counts the number of problems a team will solve based on confidence.
 *
 * This program solves the "Team" problem from Codeforces. It first reads all the
 * team members' opinions for all problems into a 2D array and then processes
 * this array to count how many problems will be implemented. A problem is
 * implemented if at least two of the three members are confident.
 *
 * @algorithm Batch Processing / Counting
 * @complexity
 * - Time: O(N), where N is the number of problems. The loops run in N*3, which is linear.
 * - Space: O(N) to store the opinions for all N problems in a 2D array.
 */
#include <bits/stdc++.h>
using namespace std;
int main(int argc, char const *argv[]) {
  // I/O optimizations for speed.
  ios_base::sync_with_stdio(false);
  cin.tie(0);
  cout.tie(0);

  int n;
  cin >> n;
  
  /**
   * @note The use of `int arr[n][3]` is a Variable Length Array (VLA), which is a
   * non-standard C++ extension. A standard C++ approach would use `std::vector`.
   */
  int arr[n][3];
  int ans = 0;

  /**
   * @brief Block Logic: Reads all inputs into the 2D array first.
   */
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 3; j++) {
      cin >> arr[i][j];
    }
  }

  /**
   * @brief Block Logic: Iterates through the stored opinions to count solvable problems.
   */
  for (int i = 0; i < n; i++) {
    int k = 0; // Counter for confident members for the current problem.
    for (int j = 0; j < 3; j++) {
      if (arr[i][j] == 1) {
        k += 1;
      }
    }

    // Pre-condition: Check if at least two members are confident.
    if (k >= 2) {
      ans++;
    }
  }
  cout << ans;

  return 0;
}
