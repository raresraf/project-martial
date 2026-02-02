/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem from Codeforces.
 * @details The program calculates the number of problems that a team of three
 * will implement. The team decides to implement a problem if at least two of
 * them are sure about the solution.
 *
 * Algorithm:
 * 1. Read the number of problems, 'n'.
 * 2. Initialize a counter for implementable problems to 0.
 * 3. Loop 'n' times:
 *    a. Read the three friends' opinions (1 for sure, 0 for not sure).
 *    b. Sum the three opinions.
 *    c. If the sum is 2 or greater, increment the counter.
 * 4. Print the final count.
 *
 * Time Complexity: O(n), where 'n' is the number of problems.
 * Space Complexity: O(1).
 */
#include <iostream>

using namespace std;

int main() {
  // n: number of problems.
  // p, v, t: opinions of the three friends (Petya, Vasya, Tonya).
  // counter: number of problems they will implement.
  int n, p, v, t, counter = 0;
  cin >> n;
  for (int i = 0; i < n; ++i) {
    cin >> p >> v >> t;
    // A problem is implemented if at least two friends are sure.
    if (p + v + t >= 2)
      counter++;
  }
  cout << counter << endl;
  return 0;
}