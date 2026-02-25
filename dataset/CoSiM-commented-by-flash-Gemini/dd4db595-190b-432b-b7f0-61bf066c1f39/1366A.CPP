/**
 * @file 1366A.CPP
 * @brief Solves Codeforces Problem 1366A, "Shovels and Swords".
 *        The problem involves finding the maximum number of items that can be crafted
 *        given two types of resources 'a' and 'b'. Crafting one item requires
 *        one of resource 'a' and two of resource 'b', or two of 'a' and one of 'b'.
 *        Alternatively, one item can be crafted using one of each resource type,
 *        but the solution implies a more efficient strategy: sum of resources / 3.
 *        The maximum number of items is the minimum of the available 'a', 'b',
 *        and the total sum divided by 3.
 * Algorithm: Greedy / Mathematical (Finding the minimum of three quantities).
 * Time Complexity: O(T), where T is the number of test cases. Each test case is O(1).
 * Space Complexity: O(1).
 */

/********************************************************************************
 *                                                                               *
 *        Author     : Rezoan Shakil Prince * Batch      : 14 * Email      :
 *prince.cuet.77@gmail.com                                  * University :
 *Chittagong University of Engineering and Technology       *
 *                                                                               *
 ********************************************************************************/

#include <bits/stdc++.h>
using namespace std;

/**
 * @brief Macro for optimizing C++ standard input/output operations.
 *        Disables synchronization with C stdio and unties cin from cout,
 *        leading to faster I/O in competitive programming.
 */
#define IOS                                                                    \
  ios::sync_with_stdio(0);                                                     \
  cin.tie(0);                                                                  \
  cout.tie(0);
#define endl "\n"

typedef long long ll;

/**
 * @brief Solves a single test case for the Shovels and Swords problem.
 *
 * Reads two quantities 'a' and 'b' representing available resources.
 * Calculates the maximum number of items that can be crafted.
 * The logic is to take the minimum of:
 *   1. Available resource 'a'.
 *   2. Available resource 'b'.
 *   3. One-third of the total combined resources (a + b) / 3.
 * This covers scenarios where one resource is abundant, or resources are balanced.
 *
 * @param None (reads input directly).
 * @return void (prints output directly).
 */
void solve() {
  ll a, b;
  // Block Logic: Processes multiple pairs of (a, b) if available in a single test case block.
  // Invariant: 'a' and 'b' contain the current resource quantities for processing.
  while (cin >> a >> b)
    // The maximum number of items is the minimum of individual resources or the average available resources per item.
    cout << min(a, min(b, (a + b) / 3)) << endl;
}

/**
 * @brief Main function of the program.
 *
 * Sets up fast I/O, reads the number of test cases, and calls the `solve`
 * function for each test case.
 *
 * @return 0 if the program executes successfully.
 */
int main() {
  // Optimize C++ standard streams for faster execution.
  IOS;

  // #ifndef prince77
  //         freopen("input.txt", "r", stdin);
  //         freopen("output.txt", "w", stdout);
  // #endif

  int tc; // Variable to store the number of test cases.
  // Read the number of test cases.
  cin >> tc;
  // Block Logic: Iterates through each test case.
  // Invariant: 'tc' decrements after each call to solve(), until all test cases are processed.
  while (tc--)
    solve(); // Call the solve function for each test case.

  return 0; // Indicate successful program termination.
}
