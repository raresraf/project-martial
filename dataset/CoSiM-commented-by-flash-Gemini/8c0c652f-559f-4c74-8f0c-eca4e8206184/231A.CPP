/**
 * @8c0c652f-559f-4c74-8f0c-eca4e8206184/231A.CPP
 * @brief This program determines the number of problems a team will implement
 * in a programming contest. For each problem, three team members express their
 * confidence (1 if confident, 0 if not). A problem is implemented if at least
 * two team members are confident. This is a concise implementation of the "A. Team" problem.
 * Algorithm: Iterative counting with conditional sum.
 * Time Complexity: O(N), where N is the number of problems.
 * Space Complexity: O(1) for storing counters and problem confidence levels.
 */

#include <iostream> // For input/output operations
using namespace std;

int main() {
  // @var n: Total number of problems.
  // @var ans: Counter for problems that will be implemented.
  int n, ans = 0;
  
  // Block Logic: Read the total number of problems.
  cin >> n;
  
  // Block Logic: Iterate through each problem to determine if it will be implemented.
  // Invariant: 'ans' accumulates the count of successfully implemented problems.
  // Pre-condition: 'n' specifies the number of problems to evaluate.
  for (int i = 0; i < n; i++) {
    // @var a, b, c: Confidence levels (1 or 0) of the three team members for a problem.
    int a, b, c;
    // Block Logic: Read confidence levels for the current problem.
    cin >> a >> b >> c;
    
    // Block Logic: Check if the sum of confidence levels is 2 or more.
    // If true, at least two team members are confident, and the problem is implemented.
    if (a + b + c >= 2)
      ans++; // Increment the count of implemented problems.
  }
  
  // Post-condition: Print the total number of problems that will be implemented.
  cout << ans << endl; // Print the result followed by a newline.
  return 0; // Explicitly return 0 for successful execution.
}
