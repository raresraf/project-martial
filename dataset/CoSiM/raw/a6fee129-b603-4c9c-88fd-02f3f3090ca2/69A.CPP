/*
    Problem link:
    http://codeforces.com/problemset/problem/80/A
*/

#include <bits/stdc++.h>

using namespace std;

bool is_prime(const int &num) {
  if (num < 2)
    return false;

  for (int factor = 2; factor * factor <= num; factor++)
    if (num % factor == 0 and num != factor)
      return false;

  return true;
}

int main() {
  ios_base::sync_with_stdio(false);

  int num1, num2;
  cin >> num1 >> num2;

  if (is_prime(num1) and is_prime(num2)) {
    bool found_prime = false;
    for (int num = num1 + 1; num < num2; num++)
      if (is_prime(num)) {
        found_prime = true;
        break;
      }

    if (found_prime)
      cout << "NO\n";
    else
      cout << "YES\n";
  } else
    cout << "NO\n";

  return 0;
}