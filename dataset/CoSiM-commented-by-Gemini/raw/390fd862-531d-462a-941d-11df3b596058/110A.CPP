/**
 * @file 110A.CPP
 * @brief Determines if a number is a "nearly lucky number".
 *
 * A "lucky number" is a positive integer whose decimal representation contains
 * only the digits 4 and 7. A "nearly lucky number" is a number where the
 * count of lucky digits (4s and 7s) in it is itself a lucky number. This
 * program checks if a given number is a nearly lucky number.
 *
 * Algorithm:
 * 1. Read a long long integer `n`.
 * 2. Count the occurrences of the digits '4' and '7' in `n`.
 * 3. Check if this count is equal to 4 or 7.
 * 4. If the count is 4 or 7, the number is "nearly lucky", so print "YES".
 * 5. Otherwise, print "NO".
 *
 * Time Complexity: O(log10(N)), where N is the input number. The program
 * iterates once for each digit of the number.
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>

using namespace std;

typedef long long ll;
typedef vector<int> vi;

/**
 * @brief Counts lucky digits and determines if the count is lucky.
 * @param n The number to check.
 * @return "YES" if the number is nearly lucky, "NO" otherwise.
 */
string solve(ll n) {
  int cnt = 0;
  /**
   * @brief Loop through each digit of the number.
   * Pre-condition: `n` is the input number.
   * Invariant: `cnt` holds the count of lucky digits from the parts of the
   * number already processed.
   */
  while (n > 0) {
    // Check if the last digit is a lucky digit (4 or 7).
    if (n % 10 == 4 || n % 10 == 7) {
      cnt++;
    }
    // Remove the last digit.
    n /= 10;
  }

  // A number is nearly lucky if the count of its lucky digits is 4 or 7.
  return (cnt == 4 || cnt == 7) ? "YES" : "NO";
}

int main() {
  // Fast I/O operations.
  ios::sync_with_stdio(false);
  cin.tie(NULL);

  ll n;
  // Read the input number.
  cin >> n;
  // Call the solve function and print the result.
  cout << solve(n) << endl;
}
