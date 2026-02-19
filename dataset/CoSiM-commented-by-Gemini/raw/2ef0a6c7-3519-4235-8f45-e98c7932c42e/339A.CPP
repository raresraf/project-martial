/**
 * @file 339A.CPP
 * @brief Solution to the "Helpful Maths" problem on Codeforces.
 * @details This program takes a string representing a sum of single-digit numbers (e.g., "3+2+1")
 * and rearranges it into non-decreasing order (e.g., "1+2+3").
 *
 * Algorithm:
 * 1. Read the entire input line, e.g., "3+2+1", into a string.
 * 2. Create a vector of integers to store the numbers from the sum.
 * 3. Iterate through the input string at even-numbered indices (0, 2, 4, ...) to access the digit characters.
 * 4. Convert each digit character to its integer equivalent and add it to the vector.
 * 5. Sort the vector of numbers in ascending order.
 * 6. Iterate through the sorted vector.
 * 7. For each number, print it. Use a ternary operator to print a '+' after the number if it's not the last one,
 *    or a newline character if it is.
 *
 * Time Complexity: O(L log L), where L is the number of digits in the sum. This is dominated by the `std::sort` operation.
 * Space Complexity: O(L), to store the numbers in a vector.
 */
#include <bits/stdc++.h>

// Custom macros for loops and vector operations, common in competitive programming.
#define FOR(i, a, n) for (int i = (int)(a); i < (int)(n); ++i)
#define REP(i, n) FOR(i, 0, n)
#define all(a) a.begin(), a.end()
#define pb push_back

typedef unsigned long long llu;
typedef long long ll;
typedef long double ld;

using namespace std;

int main() {
  string line;
  getline(cin, line);
  vector<int> num;

  /**
   * Block Logic: Extract all the numbers from the sum string.
   * Pre-condition: `line` contains the input in "d1+d2+..." format.
   * Invariant: The `num` vector contains all the digits as integers.
   */
  for (int i = 0; i < line.size(); i += 2)
    num.pb(line[i] - '0');
  
  // Sort the vector of numbers.
  sort(all(num));

  /**
   * Block Logic: Print the sorted numbers with '+' separators.
   * Pre-condition: `num` is a sorted vector of integers.
   * Invariant: The output is formatted as a sorted sum.
   */
  REP(i, num.size()) printf("%d%c", num[i], i == num.size() - 1 ? '\n' : '+');
  
  return 0;
}
