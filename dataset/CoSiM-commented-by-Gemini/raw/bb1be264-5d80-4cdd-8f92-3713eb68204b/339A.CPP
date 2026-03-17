/**
 * @file 339A.CPP
 * @brief Solution to the "Helpful Maths" problem.
 *
 * @details This program takes a string representing a sum of single digits (1, 2, or 3),
 * such as "3+2+1", and rearranges it into non-decreasing order, e.g., "1+2+3".
 *
 * @section
 * Algorithm:
 * 1. Read the input string which represents the sum.
 * 2. Create a new string containing only the digits from the input string by filtering out the '+' signs.
 * 3. Sort the new string of digits. Since the digits are single characters, a standard lexicographical
 *    sort is sufficient to order them numerically.
 * 4. Reconstruct the sorted sum by printing the digits interspersed with '+' signs.
 *
 * Time Complexity: O(L log L), where L is the number of digits in the string, dominated by the sort operation.
 * Space Complexity: O(L), to store the extracted digits.
 */
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <set>
#include <utility>
#include <vector>
#define MX 100000
#define fs first
#define sec second
#define TC                                                                     \
  int T;                                                                       \
  cin >> T;                                                                    \
  while (T--)
#define loop(i, a, c) for (int i = 0; i < a; i++)
#define loop1(i, a, c) for (int i = 1; i <= a; i++)
#define pb(a) push_back(a)
#define all(a) (a).begin(), (a).end()
#define mp(x, y) make_pair((x), (y))
#define ll long long int
using namespace std;
int main() {
  string s;
  cin >> s;
  string p = "";
  /**
   * @brief This loop extracts the numbers from the input sum.
   * Logic: It iterates through the input string 's' and appends any character
   * that is not a '+' to a new string 'p'.
   */
  for (int i = 0; i < s.size(); i++) {
    if (s[i] != '+') {
      p += s[i];
    }
  }
  // Functional Utility: Sorts the string of digits. '1'<'2'<'3', so a character sort works.
  sort(p.begin(), p.end());
  /**
   * @brief This loop reconstructs the sorted sum for printing.
   * Logic: It prints each digit from the sorted string 'p' followed by a '+',
   * ensuring the last digit is printed without a trailing '+'.
   */
  for (int i = 0; i < p.size() - 1; i++) {
    cout << p[i] << "+";
  }
  cout << p[p.size() - 1];
}
