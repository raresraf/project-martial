/**
 * @file 231A.CPP
 * @brief Solution for Codeforces Round #143, Problem 231A ("Team").
 * @details This program calculates the number of problems that will be implemented by a team of three.
 * The team decides to implement a problem only if at least two of them are sure about the solution.
 * The input consists of 'n' problems, and for each problem, three integers (0 or 1) representing
 * the confidence of each team member. The program counts how many problems have at least two members
 * confident about the solution.
 *
 * @algorithm Simple Counting
 * @complexity
 * Time: O(N), where N is the number of problems, because it iterates through each problem once.
 * Space: O(1), as it uses a fixed amount of memory regardless of the number of problems.
 */
#include <iostream>

/*
 * Contest: Code Forces Round #143 (Task 231-A)
 * URL: http://codeforces.ru/contest/231/problem/A
 */

/**
 * @struct Task
 * @brief Represents a single problem and the confidence of the three team members.
 */
struct Task {
  int a, b, c; // Confidence of the three members (1 for sure, 0 for not sure).

  /**
   * @brief Default constructor.
   */
  Task() {}

  /**
   * @brief Parameterized constructor.
   * @param _a Confidence of the first member.
   * @param _b Confidence of the second member.
   * @param _c Confidence of the third member.
   */
  Task(int _a, int _b, int _c) {
    a = _a;
    b = _b;
    c = _c;
  }

  /**
   * @brief Checks if the task is solvable.
   * @return True if at least two members are confident, false otherwise.
   */
  bool solvable(void) {
    // A task is solvable if the sum of confidences is 2 or more.
    return ((a + b + c) >= 2);
  }
};

int main(int argc, char **argv) {
  int n, count = 0;
  Task t;

  // Read the total number of problems.
  std::cin >> n;

  /**
   * @brief Main loop to process each problem.
   * Pre-condition: 'n' is the number of problems to process.
   * Invariant: After each iteration, 'count' holds the number of solvable problems found so far.
   */
  while (n--) {
    // Read the confidence levels for the current problem.
    std::cin >> t.a >> t.b >> t.c;
    // If the task is solvable, increment the counter.
    if (t.solvable())
      count++;
  }

  // Print the total number of solvable problems.
  std::cout << count << std::endl;
  return 0; // Indicate successful execution.
}