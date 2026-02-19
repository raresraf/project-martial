/**
 * @file 160A.CPP
 * @brief Solution to a competitive programming problem: "Twins".
 *
 * This problem involves partitioning a set of 'n' coins (with given values)
 * between two twins such that one twin's sum is strictly greater than the other's,
 * and the first twin takes the minimum possible number of coins.
 * This is achieved by taking the largest coins first.
 *
 * Algorithm: Greedy approach after sorting in descending order.
 * Time Complexity: O(N log N), primarily due to the sorting step.
 */
#include <algorithm> // Functional Utility: Included for the std::sort function.
#include <iostream>  // Functional Utility: Included for standard input/output operations.
using namespace std; // Functional Utility: Uses the standard namespace to avoid prefixing standard library elements with std::

/**
 * @brief Custom comparison function for sorting integers in descending order.
 *
 * This function is used with `std::sort` to arrange elements from largest to smallest.
 *
 * @param a (int): The first integer for comparison.
 * @param b (int): The second integer for comparison.
 * @return (bool): True if 'a' is greater than 'b', False otherwise.
 */
bool cmp(int a, int b) { return a > b; }

int main() {
  // Functional Utility: Optimizes standard I/O streams for faster execution,
  // crucial in competitive programming for time-sensitive tasks.
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

  // Variable Declarations:
  // n: The total number of coins.
  // c[101]: An array to store the values of the coins.
  // sum: The total sum of all coin values.
  int n, c[101], sum = 0;

  // Input: Reads the total number of coins.
  cin >> n;

  // Block Logic: Reads the values of 'n' coins into the array 'c'.
  // Invariant: After this loop, 'c' contains all coin values.
  for (int i = 0; i < n; i++)
    cin >> c[i];

  // Block Logic: Calculates the total sum of all coin values.
  // Invariant: After this loop, 'sum' holds the sum of all elements in 'c'.
  for (int i = 0; i < n; i++)
    sum += c[i];

  // Block Logic: Sorts the coin values in descending order.
  // Precondition: 'c' contains 'n' integer values.
  // Functional Utility: Uses the custom 'cmp' function to sort from largest to smallest.
  sort(c, c + n, cmp);

  // Block Logic: Adjusts the target sum to ensure one twin's sum is minimal but greater than half.
  // If the total sum is odd, half needs to be rounded up for the "strictly greater" condition.
  if (sum % 2 == 1)
    sum += 1; // Inline: Increments sum if it's odd to ensure ceiling division when divided by 2.
  sum /= 2;   // Inline: Calculates the target sum (half of total, rounded up if odd).

  // Variable Declarations for greedy selection:
  // taken: Counts the number of coins taken by the first twin.
  // inHand: Accumulates the sum of coin values taken by the first twin.
  int taken = 0, inHand = 0;

  // Block Logic: Greedily selects coins starting from the largest.
  // Invariant: `inHand` accumulates the sum of `taken` largest coins.
  for (int i = 0; i < n; i++) {
    inHand += c[i]; // Inline: Adds the current largest coin to the twin's sum.
    taken++;        // Inline: Increments the count of coins taken.

    // Block Logic: Checks if the accumulated sum exceeds the target sum.
    // Precondition: `inHand` is the sum of `taken` largest coins.
    if (inHand > sum)
      break; // Inline: Stops taking coins once the target sum is surpassed.
  }

  // Output: Prints the minimum number of coins taken to achieve the target sum.
  cout << taken << endl;
  return 0; // Functional Utility: Indicates successful program execution.
}
