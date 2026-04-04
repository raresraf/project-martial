/**
 * @file 339A.CPP
 * @brief Solves the "Helpful Maths" problem on Codeforces.
 *
 * @details The program takes a string representing a sum of single-digit numbers
 * (1, 2, or 3), such as "3+2+1", and rearranges it into non-decreasing order,
 * like "1+2+3".
 *
 * Algorithm:
 * 1. Read the input sum as a string.
 * 2. Extract the numbers from the string. The numbers are at even indices
 *    (0, 2, 4, ...), and the '+' signs are at odd indices.
 * 3. Store the extracted numbers in an integer vector.
 * 4. Sort the vector of numbers in non-decreasing order.
 * 5. Print the sorted numbers, separated by '+' signs.
 *
 * Time Complexity: O(L log L), where L is the number of digits in the sum.
 * The complexity is dominated by the sorting step.
 * Space Complexity: O(L) to store the numbers in a vector.
 */
#include <bits/stdc++.h>
using namespace std;

// Common typedefs and macros for competitive programming.
typedef long long ll;
typedef long double ld;
typedef vector<int> vi;
typedef pair<int, int> pii;
typedef pair<long long, long long> pll;
typedef vector<long long> vl;
typedef vector<pair<int, int>> vii;
#define mp make_pair
#define pb push_back
#define fr(i, j, k) for (i = j; i < k; i++)
#define FR(i, j, k) for (i = j; i <= 0; i--)
#define MOD 1000000007
#define all(v) v.begin(), v.end()

void fast() {
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
}

int main() {
  string temp;

  cin >> temp;

  vi a;
  /**
   * @brief Extract numbers from the string sum.
   * Logic: The loop iterates with a step of 2 to skip the '+' operators
   * and processes only the character digits.
   */
  for (int i = 0; i < temp.size(); i += 2) {
    a.pb(temp[i] - '0');
  }

  // Sort the numbers in non-decreasing order.
  sort(all(a));

  /**
   * @brief Print the sorted numbers in the "sum" format.
   * Logic: Print each number followed by a '+', except for the last number
   * to avoid a trailing '+'.
   */
  for (int i = 0; i < a.size() - 1; i++) {
    cout << a[i] << '+';
  }
  cout << a[a.size() - 1];
}
