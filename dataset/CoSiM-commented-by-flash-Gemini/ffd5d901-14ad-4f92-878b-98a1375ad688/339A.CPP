/**
 * @file 339A.CPP
 * @brief This program parses a string representing a sum of single-digit numbers,
 *        sorts these numbers in non-decreasing order, and then prints them
 *        back as a sum.
 * Algorithm: Parsing, Sorting.
 * Time Complexity: O(N log N) due to `std::sort`, where N is the number of digits.
 *                  Parsing and printing operations are O(N).
 * Space Complexity: O(N) for storing extracted digits in a vector.
 */
#include <bits/stdc++.h>
using namespace std;

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

/**
 * @brief Optimizes C++ standard input/output operations.
 * Functional Utility: Disables synchronization with C's stdio and unties cin from cout,
 *                     leading to faster input and output, which is crucial in
 *                     competitive programming for large datasets.
 */
void fast() {
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
}

int main() {
  string temp;

  // Reads the input string, which represents a sum of numbers (e.g., "1+2+3").
  cin >> temp;

  vi a;
  /**
   * Block Logic: Extracts single-digit numbers from the input string.
   * Invariant: Each iteration processes a digit, skipping the '+' character.
   */
  for (int i = 0; i < temp.size(); i += 2) {
    // Inline: Converts the character digit to its integer equivalent.
    a.pb(temp[i] - '0');
  }

  // Sorts the extracted digits in non-decreasing order.
  sort(all(a));

  /**
   * Block Logic: Prints the sorted digits, separated by '+' signs.
   * Invariant: All but the last digit are followed by a '+'.
   */
  for (int i = 0; i < a.size() - 1; i++) {
    cout << a[i] << '+';
  }
  // Prints the last digit without a trailing '+'.
  cout << a[a.size() - 1];
}
