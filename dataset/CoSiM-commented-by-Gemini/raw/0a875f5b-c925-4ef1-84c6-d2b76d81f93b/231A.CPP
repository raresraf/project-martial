/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem, a common task in competitive programming.
 * @details This program determines how many problems a team of three will decide to implement.
 * It first reads an integer `n`, the total number of problems. Then, for each of the `n` problems,
 * it reads three integers (0 or 1), each representing the confidence of a team member (Petya, Vasya, and Tonya).
 * If the sum of their confidences is 2 or more, it means at least two members are sure about the solution,
 * and the team will solve that problem. The program counts and prints the total number of problems they will implement.
 *
 * @algorithm Counting based on a threshold sum.
 * @complexity
 * Time complexity: O(N), where N is the number of problems, because it iterates through each problem once.
 * Space complexity: O(1), as it only uses a few variables to store input and the count.
 */
// the-Quert 231A - Team
#include <iostream>

using namespace std;

int main() {
  // n: number of problems, Petya, Vasya, Tonya: confidence of each member (0 or 1), number: counter for problems to be solved.
  int n, Petya, Vasya, Tonya, number(0);
  cin >> n;
  /**
   * @brief Loop to process each problem.
   * @invariant Iterates `n` times, processing one problem's data in each iteration.
   */
  while (n--) {
    cin >> Petya >> Vasya >> Tonya;
    /**
     * @brief Check if the team will solve the problem.
     * @pre The confidences of Petya, Vasya, and Tonya have been read.
     * @post The `number` counter is incremented if the sum of confidences is at least 2.
     */
    if (Petya + Vasya + Tonya >= 2) {
      number += 1;
    }
  }
  // Print the total number of problems to be solved.
  cout << number << endl;
  return 0;
}