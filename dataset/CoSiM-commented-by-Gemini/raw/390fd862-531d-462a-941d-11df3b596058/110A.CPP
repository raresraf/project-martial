/**
 * @file 110A.CPP
 * @brief Solution to the "Nearly Lucky Number" problem (Codeforces 110A).
 *
 * @details A "nearly lucky number" is defined as a number where the count of its
 * "lucky digits" (which are 4 and 7) is itself a lucky number. This program
 * checks if a given number fits this criterion.
 *
 * @algorithm The core logic is implemented in the `solve` function.
 * 1. It iterates through each digit of the input number `n` by repeatedly
 *    taking the number modulo 10 and then dividing by 10.
 * 2. It maintains a counter `cnt` which is incremented for each digit that is
 *    either a 4 or a 7.
 * 3. After counting the lucky digits, it checks if the count `cnt` is itself
 *    a lucky number (i.e., if `cnt` is 4 or 7).
 * 4. It returns "YES" if the count is lucky, and "NO" otherwise.
 *
 * @complexity
 * Time Complexity: O(log10(N)), where N is the input number. The runtime is
 *                  proportional to the number of digits in N.
 * Space Complexity: O(1), as it uses a fixed amount of space for variables.
 */
#include <bits/stdc++.h>

using namespace std;

typedef long long ll;
typedef vector<int> vi;

/**
 * @brief Counts lucky digits (4 or 7) and checks if the count is lucky.
 * @param n The input number to check.
 * @return "YES" if the number is nearly lucky, "NO" otherwise.
 */
string solve(ll n) {
  int cnt = 0;
  // Block Logic: Iterate through each digit of the number.
  while (n > 0) {
    if (n % 10 == 4 || n % 10 == 7) {
      cnt++;
    }
    n /= 10;
  }

  // Block Logic: Check if the count of lucky digits is itself a lucky number (4 or 7).
  return (cnt == 4 || cnt == 7) ? "YES" : "NO";
}

int main() {
  // Functional Utility: Decouples C++ streams from C streams for faster I/O.
  ios::sync_with_stdio(false);
  cin.tie(NULL);

  ll n;
  cin >> n;
  cout << solve(n) << endl;
}
