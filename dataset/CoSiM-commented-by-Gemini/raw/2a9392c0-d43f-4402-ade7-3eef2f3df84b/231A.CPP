/**
 * @file 231A.CPP
 * @brief Solution to the "Team" problem (231A) from Codeforces.
 * @details The program counts the number of problems a three-person team will solve,
 * where a problem is solved if at least two members are confident (represented by a '1').
 *
 * @author Your Name (or leave blank)
 *
 * @algorithm The solution iterates through each problem, summing the confidence
 * levels of the team members to see if it meets the threshold of 2 or more.
 * 1. Read the total number of problems (`num_tasks`).
 * 2. Initialize a counter `solutions` to 0.
 * 3. Loop `num_tasks` times.
 * 4. In each iteration, read the three confidence levels (`petya`, `vasya`, `tonya`).
 * 5. Use a ternary operator to add 1 to `solutions` if the sum of confidences is
 *    greater than 1, otherwise add 0.
 * 6. After the loop, print the final count of solutions.
 *
 * @complexity
 * Time Complexity: O(N), where N is the number of problems, due to the single loop.
 * Space Complexity: O(1), as it uses a fixed amount of memory.
 */
#include <iostream>

using namespace std;

int main() {
  int num_tasks;
  cin >> num_tasks;

  int solutions{0}; // Use of brace-initialization.
  /**
   * @block
   * @brief Processes each of the 'num_tasks' problems to count solvable ones.
   * @invariant 'solutions' holds the count of problems solved so far.
   * @pre 'num_tasks' is the total number of problems.
   * @post 'solutions' holds the total count where at least two members are confident.
   */
  for (int num_task = 0; num_task < num_tasks; ++num_task) {
    int petya, vasya, tonya;
    cin >> petya >> vasya >> tonya;
    // A ternary operator concisely adds 1 to the count if the sum of opinions is > 1.
    solutions += (petya + vasya + tonya) > 1 ? 1 : 0;
  }
  cout << solutions << endl;
  return 0;
}
