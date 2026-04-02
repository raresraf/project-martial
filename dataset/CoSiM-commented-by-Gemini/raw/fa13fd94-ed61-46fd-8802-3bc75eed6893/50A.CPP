/**
 * @file 50A.CPP
 * @brief Solves a problem resembling "Team" (231A) from Codeforces.
 * @details This program reads an integer 'd', followed by 'd' rows of three integers.
 *          It counts the number of rows where the sum of the three integers is at least 2.
 *
 * Algorithm: Simple iterative counting based on a condition.
 * Time Complexity: O(d), where 'd' is the number of input rows.
 * Space Complexity: O(1).
 */
#include <iostream>
using namespace std;
int main() {
  long long int a, b, c, d, coun = 0;
  ;
  cin >> d;
  /**
   * @brief Iterates through each of the 'd' test cases.
   */
  for (int i = 1; i <= d; i++) {
    cin >> a;
    cin >> b;
    cin >> c;
    // Condition: Checks if at least two of the three inputs are non-zero (or sum to >= 2).
    if (a + b + c >= 2)
      coun++;
  }
  // Prints the final count. Note the post-increment `coun++` will print the value before incrementing.
  cout << coun++;
}
