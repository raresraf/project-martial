/**
 * @file 231A.CPP
 * @brief Solution to Codeforces Problem 231A: "Team".
 *
 * This problem involves determining how many problems a team of three friends
 * will implement. The team implements a problem if at least two of them are
 * sure about the solution.
 *
 * Algorithm: Counting sure friends per problem.
 * For each problem, the program reads three integers representing the friends'
 * certainty (1 if sure, 0 if not sure). It then sums these values. If the sum
 * is 2 or more, the problem is implemented, and a counter is incremented.
 *
 * Time Complexity: O(N), where N is the number of problems, as it involves
 *                  reading N sets of three integers and performing constant
 *                  time operations per set.
 * Space Complexity: O(N) for storing the input data in a 2D array.
 */
#include <iostream> // Required for input/output operations (cin, cout).

using namespace std; // Uses the standard namespace to avoid prefixing std::

#define long long ll; // Defines 'll' as a shorthand for 'long long'. (Unused in this specific problem).

int n; // Global variable to store the number of problems.
int arr[1000][3]; // Global 2D array to store the certainty of three friends for up to 1000 problems.

/**
 * @brief Solves the "Team" problem by iterating through stored problem data
 *        and counting how many problems meet the implementation criteria.
 * Prints the final count to standard output.
 */
void solve() {
  int answer = 0; // Local variable to count the problems implemented.
  /**
   * Block Logic: Iterates through each problem's data stored in the global 'arr'.
   * Pre-condition: 'arr' is populated with problem data, 'n' is the number of problems.
   * Invariant: 'answer' accurately reflects the count of problems with at least two
   *            sure friends encountered so far.
   */
  for (int *i : arr) { // C++11 range-based for loop. 'i' is a pointer to an array of 3 integers.
    // Block Logic: Checks if the sum of certainties for the current problem is greater than 1.
    // This indicates at least two friends are sure.
    if (i[0] + i[1] + i[2] > 1)
      answer++; // Increments the counter if the problem will be implemented.
  }
  cout << answer; // Prints the total number of problems implemented.
}

/**
 * @brief Main function to read input, call the solver function, and manage execution flow.
 *
 * @return int Returns 0 on successful execution.
 */
int main() {
  cin >> n; // Reads the number of problems.
  /**
   * Block Logic: Reads the certainty of three friends for each of the 'n' problems.
   * Pre-condition: 'n' specifies the number of problems to read.
   * Invariant: The 'arr' 2D array is populated with 'n' rows of problem data.
   */
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 3; j++) {
      cin >> arr[i][j]; // Reads the certainty for each friend for the current problem.
    }
  }
  solve(); // Calls the solve function to process the data and print the result.
}