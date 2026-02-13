/**
 * @file 339A.CPP
 * @brief Solution for the "Helpful Maths" problem (Codeforces 339A).
 * @author competitive-programmer
 *
 * @details This program reorders a sum of numbers (1, 2, and 3) given as a
 * string into non-decreasing order. For example, an input of "3+2+1"
 * should produce the output "1+2+3".
 *
 * Algorithm:
 * 1. Read the input string `s` representing the sum.
 * 2. Create a temporary string `temp` to hold only the numbers from the sum.
 * 3. Iterate through the input string `s` with a step of 2 to extract the
 *    digit characters, and append them to `temp`.
 * 4. Use `std::sort` on the `temp` string. Since characters '1', '2', '3' have
 *    lexicographical order corresponding to their numerical order, this
 *    correctly sorts the numbers.
 * 5. Iterate through the original string `s` again with a step of 2 and
 *    replace the digit characters with the sorted ones from `temp`.
 * 6. Print the modified string `s`.
 *
 * Time Complexity: O(N log N) due to the sort operation, where N is the number
 * of digits in the sum.
 * Space Complexity: O(N) to store the temporary string of digits.
 */
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// The following variables are part of a template and are not used in this solution.
const int sz = 100010;
vector<ll> v(sz, 0), prefix(sz, 0), suffix(sz, 0);
string s;

int main() {
  int n;
  string s, temp = "";
  cin >> s;

  /**
   * Block Logic: Extract the digits from the input string.
   * The loop iterates with a step of 2 to skip the '+' signs.
   */
  for (int i = 0; i < s.length(); i += 2) {
    temp.push_back(s[i]);
  }

  // Sort the string of digits. '1'<'2'<'3', so character sort works.
  sort(temp.begin(), temp.end());

  /**
   * Block Logic: Reconstruct the sorted sum string.
   * The loop iterates through the positions of the digits in the original
   * string and replaces them with the sorted digits from `temp`.
   */
  int k = 0; // Index for the sorted `temp` string.
  for (int i = 0; i < s.length(); i += 2, ++k) {
    s[i] = temp[k];
  }

  // Print the final, sorted sum.
  cout << s;
  return 0;
}
