
/**
 * @file 160A.CPP
 * @brief Solves a greedy problem: find minimum number of largest elements whose sum is greater than half of total sum.
 *
 * This program reads an integer 'n' followed by 'n' integers. The goal is to
 * find the minimum number of these integers, starting from the largest, whose
 * sum is strictly greater than the sum of the remaining integers. This is a
 * classic greedy algorithm application.
 *
 * Algorithm: Greedy approach. Sort the numbers in ascending order. Then, iterate
 *            from the largest numbers, accumulating their sum (`sum2`) while
 *            subtracting them from the total sum (`sum1`). Stop when `sum2`
 *            exceeds `sum1`.
 * Time Complexity: O(N log N), where N is the number of integers, dominated by the sorting step.
 * Space Complexity: O(N) for storing the integers in a vector.
 */
/* Monil Soni sonil06 */
/* DA-IICT  */

#include <bits/stdc++.h> // Includes most standard library headers.

using namespace std;

typedef long long LL;     // Functional Utility: Defines 'LL' as a shorthand for 'long long'.
typedef vector<int> vi;   // Functional Utility: Defines 'vi' as a shorthand for 'vector<int>'.
typedef vector<long long> vl; // Functional Utility: Defines 'vl' as a shorthand for 'vector<long long>'.

#define MOD 1000000007 // Functional Utility: Defines 'MOD' as a common modulo value.
// Functional Utility: Defines '_for(i, n)' as a shorthand for a for-loop from 0 to n-1.
#define _for(i, n) for (int i = 0; i < n; i++)
// Functional Utility: Defines '__for(i, n)' as a shorthand for a for-loop from 1 to n.
#define __for(i, n) for (int i = 1; i <= n; i++)
// Functional Utility: Defines 'REP(i, s, n)' as a shorthand for a for-loop from 's' to 'n'.
#define REP(i, s, n) for (int i = (s), _n = (n); i <= _n; i++)
// Functional Utility: Defines 'FOR(i, n)' as a shorthand for a for-loop from 0 to n-1.
#define FOR(i, n) for (int i = 0, _n = (n); i < _n; i++)
// Functional Utility: Defines 'REPD(i, e, s)' as a shorthand for a reverse for-loop from 'e' to 's'.
#define REPD(i, e, s) for (int i = (e), _s = (s); i >= _s; i--)
// Functional Utility: Defines 'tr(container, it)' as a shorthand for a type-safe iterator loop.
#define tr(container, it)                                                      \
  for (typeof(container.begin()) it = container.begin();                       \
       it != container.end(); it++)
// Functional Utility: Defines 'pb' as a shorthand for 'push_back'.
#define pb push_back
// Functional Utility: Defines 'mp' as a shorthand for 'make_pair'.
#define mp make_pair
// Functional Utility: Defines 'fi' as a shorthand for 'first' for pairs.
#define fi first
// Functional Utility: Defines 'se' as a shorthand for 'second' for pairs.
#define se second
// Functional Utility: Defines 'ALL(x)' to get iterators for the entire range of a container 'x'.
#define ALL(x) x.begin(), x.end()

int main() {
  /**
   * @brief Main function: Reads integers, sorts them, and finds the minimum count of largest elements
   *        whose sum is greater than the sum of the remaining elements.
   * @return 0 if the program executes successfully.
   */
  int n; // Number of integers.
  int x; // Temporary variable for reading integers.
  int sum1 = 0; // Stores the total sum of all integers, then sum of remaining integers.
  int sum2 = 0; // Stores the sum of selected largest integers.
  vector<int> v; // Vector to store the input integers.
  scanf("%d", &n); // Read the number of integers.
  /**
   * @brief Loop to read integers and calculate their total sum.
   * @pre 'n' is the number of integers to read.
   * @post 'v' contains all input integers, 'sum1' holds their total sum.
   * @invariant 'i' iterates from 0 to n-1.
   */
  for (int i = 0; i < n; i++) {
    scanf("%d", &x); // Read each integer.
    v.pb(x);         // Add the integer to the vector.
    sum1 += x;       // Add to the total sum.
  }
  // Functional Utility: Sorts the vector 'v' in ascending order.
  // This is a prerequisite for the greedy approach, enabling selection of largest elements.
  sort(v.begin(), v.end());
  int i; // Loop counter, also used to determine the count of selected elements.
  /**
   * @brief Greedy loop to select the minimum number of largest elements.
   * @pre 'v' is sorted. 'sum1' is total sum, 'sum2' is 0.
   * @post 'sum2' will hold the sum of selected largest elements, and 'sum1'
   *       will hold the sum of the remaining elements.
   * @invariant 'i' iterates from the largest element downwards. Loop continues as long as 'sum2'
   *            is not greater than 'sum1'.
   */
  for (i = v.size() - 1; i >= 0; i--) { // Iterate from the largest element.
    sum1 -= v[i]; // Subtract the current largest element from sum1 (remaining elements).
    sum2 += v[i]; // Add the current largest element to sum2 (selected elements).
    /**
     * @brief Break condition: if the sum of selected elements ('sum2') exceeds the sum of remaining elements ('sum1').
     */
    if (sum2 > sum1) {
      break; // Exit the loop as the condition is met.
    }
  }
  // Functional Utility: The number of selected elements is (total_elements - i).
  // 'i' is the index of the last selected element.
  printf("%d\n", (int)v.size() - i); // Output the count of selected elements.
  return 0;
}
