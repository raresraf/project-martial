/**
 * @file 231A.CPP
 * @brief Solution for a competitive programming problem (Codeforces "Team").
 * This program counts the number of problems a three-person team will solve.
 * A problem is solved if at least two of the three members are confident
 * (represented by a '1').
 *
 * @algorithm Simulation with Online Processing. The program reads the data for
 * each problem, decides if it's solvable, and updates a counter on the fly
 * without storing the entire input array.
 *
 * @time_complexity O(N), where N is the number of problems, due to the single loop.
 * @space_complexity O(1).
 */
#include <iostream>

using namespace std;

int main() {
  int n; // The total number of problems
  int count;
  cin >> n;
  count = 0; // Counter for the number of solvable problems

  /**
   * @block
   * @brief Processes each problem to see if it will be solved.
   *
   * The loop iterates `n` times. For each problem, it checks if at least two
   * members are confident. This is done by explicitly checking all pairs of two.
   * If the condition is met, the counter is incremented.
   */
  for (int i = 0; i < n; i++) {
    int a, b, c; // Confidence levels for the three members
    cin >> a >> b >> c;
    if ((a == 1 && b == 1)) {
      count = count + 1;
    } else if ((a == 1 && c == 1)) {
      count = count + 1;
    } else if ((b == 1 && c == 1)) {
      count = count + 1;
    }
  }
  cout << count << "\n";
  return 0;
}