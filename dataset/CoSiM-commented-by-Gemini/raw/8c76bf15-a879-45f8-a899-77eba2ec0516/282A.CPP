/**
 * @file 282A.CPP
 * @brief Calculates a product based on a custom mapping of input digits.
 *
 * @warning The filename '282A.CPP' is misleading. This code does not solve the
 * "Bit++" problem (282A). Instead, it solves a problem requiring a custom
 * digit-based calculation.
 *
 * @details
 * The program uses a hard-coded map where each digit from 0 to 9 corresponds
 * to a specific integer value. It reads a two-digit string, looks up the
 * corresponding integer for each of the two digits, and prints their product.
 * The specific meaning of the hard-coded values is defined by the original
 * problem statement, which is not available here.
 *
 * @section
 * Algorithm: Lookup table and direct calculation.
 * Time Complexity: O(1).
 * Space Complexity: O(1).
 */
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <vector>
using namespace std;
// Typedefs from a template, unused in this specific solution.
typedef vector<int> vi;
typedef vector<vi> vvi;
// A global vector used as a lookup table for a custom digit-to-value mapping.
vi digit(10);

int main() {

  /**
   * @brief Initialization of the custom digit-to-value map.
   * The meaning of these values is specific to the problem being solved.
   */
  digit[0] = 2;
  digit[1] = 7;
  digit[2] = 2;
  digit[3] = 3;
  digit[4] = 3;
  digit[5] = 4;
  digit[6] = 2;
  digit[7] = 5;
  digit[8] = 1;
  digit[9] = 2;

  string s;
  cin >> s;
  /**
   * @brief Performs the calculation and prints the result.
   *
   * Logic:
   * 1. The first character of the string `s` (e.g., '1') is converted to an integer (1).
   * 2. This integer is used as an index to get a value from the `digit` map (e.g., digit[1] -> 7).
   * 3. The same is done for the second character of the string.
   * 4. The two retrieved values are multiplied, and the product is printed.
   */
  cout << digit[(s[0] - '0')] * digit[(s[1] - '0')] << endl;
}
