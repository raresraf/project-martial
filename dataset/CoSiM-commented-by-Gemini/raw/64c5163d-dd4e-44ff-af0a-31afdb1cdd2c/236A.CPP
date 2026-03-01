/**
 * @file 236A.CPP
 * @brief Solution for the "Boy or Girl" problem.
 *
 * @problem The program determines if a user is male or female based on their username.
 * If the number of distinct characters in the username is odd, the user is male
 * ("IGNORE HIM!"). If the number is even, the user is female ("CHAT WITH HER!").
 *
 * @algorithm This code uses a brute-force O(N^2) approach to count distinct characters.
 * 1. It iterates through the string with an outer loop (from `i=0` to `N-1`).
 * 2. For each character `a[i]`, it uses an inner loop (from `j=0` to `i-1`) to check
 *    if `a[i]` has appeared before.
 * 3. A counter `pas` is used to track uniqueness.
 *
 * @implementation_note The logic is confusing. `pas` is initialized to 1, and the
 * final parity check is reversed (`pas % 2 == 0` for the male case). This
 * combination happens to work out correctly: if the number of distinct characters
 * is `D`, the final value of `pas` will be `D+1`, so `(D+1) % 2 == 0` is equivalent
 * to `D % 2 != 0`.
 *
 * @time_complexity O(N^2), where N is the string length, due to the nested loops.
 * @space_complexity O(N) to store the input string.
 */
#include <bits/stdc++.h>
#include <string>
#define ll long long
using namespace std;

int main() {

#ifndef ONLINE_JUDGE
  // Redirects I/O to files for local testing.
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif
  string a;
  // `pas` is a confusing counter for distinct characters, initialized to 1.
  int pas = 1, num;
  cin >> a;
  // Outer loop iterates through each character.
  for (int i = 0; i < a.size(); i++) {
    // Inner loop checks for previous occurrences of the character a[i].
    for (int j = 0; j < i; j++) {
      if (a[i] == a[j]) {
        num++;
      }
    }
    // If num is 0, the character is new/unique.
    if (num == 0) {
      pas++;
    }
    // Reset temporary counter for the next character.
    num = 0;
  }
  // The parity check logic is reversed, but works due to `pas` starting at 1.
  // If distinct chars `D` is odd, `pas` is `D+1` (even).
  // If `D` is even, `pas` is `D+1` (odd).
  if (pas % 2 == 0) {
    cout << "IGNORE HIM!";
  } else {
    cout << "CHAT WITH HER!";
  }
  return 0;
}
