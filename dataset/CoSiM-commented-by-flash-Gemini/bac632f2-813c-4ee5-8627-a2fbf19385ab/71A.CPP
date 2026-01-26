// 71A Way Too Long Words
// https://codeforces.com/problemset/problem/71/A

#include <bits/stdc++.h>

using namespace std;

int main() {
  int t;
  cin >> t;
  while (t--) {
    string str;
    cin >> str;
    int n = str.length();
    if (n <= 10) {
      cout << str << '\n';
    } else {
      cout << str[0] << n - 2 << str[n - 1] << '\n';
    }
  }
  return 0;
}