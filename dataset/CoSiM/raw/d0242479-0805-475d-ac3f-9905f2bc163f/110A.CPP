/*
 * Problem description:
 *
 * https://codeforces.com/problemset/problem/59/A
 *
 */

#include <bits/stdc++.h>

using namespace std;

int main() {
  long long n;
  int count(0);
  string a;
  cin >> n;
  a = to_string(n);
  for (char c : a) {
    if (c == '7' || c == '4') {
      count++;
    }
  }
  a = count == 7 || count == 4 ? "YES" : "NO";
  cout << a << '\n';
}
