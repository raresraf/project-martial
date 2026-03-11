/**
 * @file 231A.CPP
 * @brief Counts the number of problems that will be implemented based on team confidence.
 *
 * This program determines how many problems, out of a given set, a team of three
 * will decide to implement. The condition for implementation is that at least two
 * out of the three members are confident about the solution. Each problem's confidence
 * level is represented by three boolean-like integers (0 or 1), where 1 means a
 * member is confident.
 *
 * @algorithm The program iterates through each problem. For each problem, it reads the
 * confidence levels of the three members and calculates their sum. If the sum is
 * greater than or equal to 2, it means at least two members are confident, and a
 * counter is incremented. This is a direct and efficient implementation of the logic.
 *
 * @time_complexity O(N), where N is the number of problems, as it involves a single
 * loop through all the problems.
 * @space_complexity O(1), as it only uses a few variables to store the counts and
 * intermediate values.
 */
#include <iostream>
using namespace std;

int main() {
  int n; // The total number of problems to evaluate.
  cin >> n;

  int i = 0; // Loop counter.
  int Z = 0; // Counter for the number of problems to be implemented.
  int a, b, c; // Confidence levels of the three team members.

  /**
   * @brief Main loop to process each problem.
   * @invariant `Z` holds the count of implementable problems for the first `i` problems.
   */
  while (i < n) {
    // Read the confidence levels for the current problem.
    cin >> a >> b >> c;
    
    /**
     * @brief Check if at least two members are confident by summing their votes.
     * @pre a, b, and c are either 0 or 1.
     */
    if (a + b + c >= 2) {
      Z++; // Increment the counter if the condition is met.
    }

    i++;
  }

  // Print the total count of problems to be implemented.
  cout << Z;
}
