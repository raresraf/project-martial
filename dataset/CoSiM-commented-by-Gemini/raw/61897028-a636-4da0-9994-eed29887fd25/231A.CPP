/**
 * @file 231A.CPP
 * @brief Solution for the "Team" problem.
 *
 * @problem A team of three programmers decides to solve 'n' problems. For each
 * problem, they will only attempt it if at least two of them are sure about the
 * solution. Given the decisions for each problem (1 if sure, 0 if not), the
 * task is to count how many problems the team will solve.
 *
 * @algorithm The program iterates 'n' times, processing one problem's data in each
 * iteration. A helper function, `implement()`, encapsulates the core logic,
 * returning true if at least two of its three arguments are non-zero. The main
 * loop calls this function for each problem and increments a counter if it returns
 * true. This approach processes data as it's read, making it space-efficient.
 *
 * @time_complexity O(N), where N is the number of problems, due to the single loop.
 * @space_complexity O(1), as inputs are not stored in a collection.
 */
#include <iostream>
#include <regex>
#include <string>

/**
 * @brief Determines if a problem should be implemented based on team members' confidence.
 * @param a Decision of the first member (1 for yes, 0 for no).
 * @param b Decision of the second member (1 for yes, 0 for no).
 * @param c Decision of the third member (1 for yes, 0 for no).
 * @return True if at least two members are confident, False otherwise.
 */
bool implement(int a, int b, int c) {
  // Logic: The problem is implemented if (a AND b) OR (a AND c) OR (b AND c) are sure.
  if (a && b || a && c || b && c) {
    return true;
  } else {
    return false;
  }
}

int main() {
  int n, a, b, c;

  // Counter for the number of problems that will be implemented.
  int implementCounter = 0;

  std::cin >> n;

  // Loop through each of the 'n' problems.
  for (int i = 0; i < n; i++) {
    // Read the three decisions for the current problem.
    std::cin >> a >> b >> c;

    // Check if the problem meets the implementation criteria.
    if (implement(a, b, c)) {
      implementCounter++;
    }
  }

  std::cout << implementCounter << std::endl;

  return 0;
}
