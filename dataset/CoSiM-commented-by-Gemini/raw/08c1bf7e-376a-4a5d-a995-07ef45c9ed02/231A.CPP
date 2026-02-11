/**
 * @file 231A.CPP
 * @brief Counts the number of problems a team will solve based on member confidence.
 * @details This program solves the "Team" problem on Codeforces. A team of three
 * decides to solve a problem only if at least two of them are confident. This
 * program reads the confidence levels for a series of problems and counts how
 * many they will implement.
 *
 * @see https://codeforces.com/problemset/problem/231/A
 *
 * Algorithm:
 * 1. Read the total number of problems to consider.
 * 2. Initialize a counter for solvable problems to zero.
 * 3. Loop for each problem:
 *    a. Read the confidence level (1 for confident, 0 for not) for each of the three members.
 *    b. Sum the three confidence levels.
 *    c. If the sum is 2 or greater, increment the counter.
 * 4. Print the final value of the counter.
 *
 * Time Complexity: O(N), where N is the number of problems, because it involves a single loop
 * through all the problems.
 * Space Complexity: O(1), as it uses a fixed amount of memory regardless of the number of problems.
 */
#include <iostream>

using namespace std;

int main() {
  // problem: The total number of problems the team is considering.
  int problem;
  // Petya, Vasya, Tonya: The confidence levels of the three team members.
  int Petya = 0;
  int Vasya = 0;
  int Tonya = 0;
  // counter: The number of problems the team will implement.
  int counter = 0;
  cin >> problem;
  
  /**
   * @brief Main loop to iterate through each problem.
   *
   * Pre-condition: 'problem' holds the number of problems to evaluate.
   * Invariant: After each iteration, 'counter' holds the number of problems solved so far.
   */
  for (int i = 0; i < problem; i++) {
    cin >> Petya;
    cin >> Vasya;
    cin >> Tonya;
    
    // Sum the confidence levels of the team members.
    int temp = Petya + Vasya + Tonya;
    
    // A problem is implemented if at least two members are confident.
    if (temp >= 2) {
      counter++;
    }
  }
  
  // Output the total count of problems to be implemented.
  cout << counter;
  return 0;
}
