/**
 * @file 231A.CPP
 * @brief Solution for Codeforces Problem 231A - "Team".
 *
 * This program counts the number of problems a team will decide to implement.
 * The decision rule is that at least two out of three members must be sure.
 * The input consists of 'n' problems, each with three opinions represented
 * by 1 (sure) or 0 (not sure). The variable names are in Indonesian.
 *
 * @algorithm Simple counting and summation.
 * @time_complexity O(N), where N is the number of problems, as it processes
 *                  each problem's opinions in a single pass.
 * @space_complexity O(1).
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  // `soal` means problem, `yakin` means sure, `solusi` means solution.
  int soal, yakin = 0, solusi = 0;
  cin >> soal;
  /**
   * @brief Processes each problem.
   * This outer loop iterates through each problem to be considered.
   */
  for (int i = 0; i < soal; i++) {
    int input;
    /**
     * @brief Reads opinions and counts sure members.
     * This inner loop reads the three opinions for the current problem.
     * `yakin` (sure) is incremented for each '1' encountered.
     */
    for (int i = 0; i < 3; i++) {
      cin >> input;
      if (input == 1) {
        yakin++;
      }
    }
    /**
     * @brief Checks if the problem will be solved.
     * If the number of sure members is 2 or more, the solution count `solusi`
     * is incremented.
     */
    if (yakin >= 2) {
      solusi++;
    }
    // Reset the counter for the next problem.
    yakin = 0;
  }
  // Print the total number of solutions.
  cout << solusi;

  return 0;
}