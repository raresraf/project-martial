/**
 * @file raw/c7645f1a-e374-4e8e-9306-a6bf2e5a4273/339A.CPP
 * @brief Rearranges a sum of single-digit numbers into non-decreasing order.
 *
 * This program takes a string representing a sum of single-digit numbers
 * (e.g., "3+1+2") and outputs the sum with the numbers rearranged in
 * non-decreasing order (e.g., "1+2+3"). This implementation achieves the
 * result using direct string manipulation.
 *
 * @algorithm
 * 1. Read the input string containing the sum.
 * 2. Parse the input string by iterating with a step of 2 to extract only the
 *    digit characters, appending them to a new temporary string.
 * 3. Sort the temporary string lexicographically. Since the characters are all
 *    digits, this correctly sorts them by their numeric value.
 * 4. Print the first digit from the sorted string.
 * 5. Iterate through the rest of the sorted string, printing each digit
 *    preceded by a '+' sign to reconstruct the sum.
 *
 * @time_complexity O(N log N), where N is the number of summands. The complexity
 *                  is dominated by sorting the string of digits.
 * @space_complexity O(N) to store the temporary string containing the digits.
 */
#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
typedef pair<int, int> ii;

int main() {
  // Fast I/O setup
  cin.tie(0);
  cout.tie(0);
  ios_base::sync_with_stdio(0);
  string s;
  cin >> s;
  string t;
  /**
   * @block
   * Extracts digit characters from the input sum `s` into a new string `t`.
   */
  for (int i = 0; i < s.size(); i += 2)
    t += s[i];
  // Sorts the string of digits. "312" becomes "123".
  sort(t.begin(), t.end());
  /**
   * @block
   * Reconstructs and prints the sorted sum.
   */
  cout << t[0];
  for (int i = 1; i < t.size(); i++)
    cout << "+" << t[i];
  cout << endl;
  return 0;
}
