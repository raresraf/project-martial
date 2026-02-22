/**
 * @7284b0d5-ff92-4a59-9159-0aaefa80a8e6/231A.CPP
 * @brief Solution for a competitive programming problem to count problems with majority consensus.
 *
 * This program reads the number of problems and for each problem, the opinions of three friends
 * (1 if sure, 0 if unsure). It then determines how many problems have at least two friends
 * who are sure about the solution, indicating a consensus to implement.
 *
 * Algorithm:
 * 1. Read the total number of problems (n).
 * 2. Initialize a counter for problems with consensus to 0.
 * 3. Loop 'n' times, once for each problem:
 *    a. Read the opinions of the three friends (a, b, c).
 *    b. Count how many friends are sure (opinion is 1).
 *    c. If the count of sure friends is 2 or more, increment the consensus counter.
 * 4. Print the final count of problems with consensus.
 *
 * Time Complexity: O(N) where N is the number of problems, as each problem requires constant time processing.
 * Space Complexity: O(1) as only a few variables are used to store counts and opinions.
 */

#include <iostream> // Functional Utility: Provides input/output functionalities (std::cin, std::cout).
using namespace std;

int main() {
  int n;
  // Functional Utility: Reads the total number of problems.
  cin >> n;
  // Functional Utility: Counter to store the number of problems where at least two friends are sure.
  int count = 0;
  // Block Logic: Iterates through each problem to read opinions and check for consensus.
  // Pre-condition: 'n' problems will follow in the input, each with three opinions.
  // Invariant: 'count' accumulates problems with at least two 'sure' opinions.
  for (int i = 0; i < n; i++) {
    int a, b, c;
    // Functional Utility: Reads the opinions of three friends for the current problem.
    cin >> a >> b >> c;
    // Functional Utility: Counter for the number of friends who are sure about the current problem.
    int c1 = 0;
    // Conditional Logic: Checks if friend 'a' is sure.
    if (a == 1) {
      c1++;
    }
    // Conditional Logic: Checks if friend 'b' is sure.
    if (b == 1) {
      c1++;
    }
    // Conditional Logic: Checks if friend 'c' is sure.
    if (c == 1) {
      c1++;
    }
    // Conditional Logic: Checks if at least two friends are sure about the current problem.
    if (c1 >= 2) {
      count++; // Functional Utility: Increments the total count of problems with consensus.
    }
  }
  // Functional Utility: Prints the final count of problems with at least two sure friends, followed by a newline.
  cout << count << endl;
}
