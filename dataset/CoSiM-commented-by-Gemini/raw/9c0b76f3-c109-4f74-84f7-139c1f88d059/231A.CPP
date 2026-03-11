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
 * confidence levels of the three members. It uses a map to count the number of
 * '1's (confident members). If the count of '1's is two or more, a total counter
 * is incremented. The map is cleared after each problem.
 *
 * @note The use of `std::map` to count the number of '1's for each problem is functionally
 * correct but highly inefficient for this specific task. A simple summation of the three
 * confidence values (`a + b + c`) and checking if the sum is greater than or equal to 2
 * would achieve the same result with better performance and less memory usage.
 *
 * @time_complexity O(N * log(K)), where N is the number of problems and K is the number
 * of unique keys in the map (which is at most 2: 0 and 1). The log factor comes from
 * the map operations. A more optimal solution would be O(N).
 * @space_complexity O(1) because the map is cleared in each iteration, so it doesn't
 * grow with the number of problems.
 */
#include <bits/stdc++.h>
using namespace std;
map<int, int> m; // A map to count occurrences of confidence levels (0s and 1s).

int main() {
  int n; // The total number of problems to evaluate.
  int tot = 0; // Counter for the number of problems to be implemented.
  
  // Read the number of problems.
  cin >> n;

  /**
   * @brief Main loop to process each problem.
   * @invariant `tot` holds the count of implementable problems processed so far.
   */
  while (n--) {
    int a, b, c; // Confidence levels of the three team members.
    cin >> a >> b >> c;

    // Use the map to count how many members are confident (value is 1).
    m[a]++;
    m[c]++;
    m[b]++;
    
    /**
     * @brief Check if at least two members are confident.
     * @pre The map `m` contains the counts of 0s and 1s for the current problem.
     */
    if (m[1] >= 2)
      tot++; // Increment the counter if the condition is met.
    
    // Clear the map to prepare for the next problem.
    m.clear();
  }
  
  // Print the total count of problems to be implemented.
  cout << tot << "\n";
}
