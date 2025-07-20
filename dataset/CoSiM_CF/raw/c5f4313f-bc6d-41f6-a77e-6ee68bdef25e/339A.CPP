#include <bits/stdc++.h>

using namespace std;

#define ll long long

int main() {
  bool flag = false;
  string s;
  cin >> s;
  int n[s.length() / 2 + 1];
  for (int i = 0; i < s.length(); i += 2) {
    n[i / 2] = s[i];
  }
  sort(n, n + (s.length() / 2 + 1));
  for (int i = 0; i < s.length() / 2 + 1; i++) {
    if (flag)
      cout << "+";
    flag = true;
    cout << n[i] - 48;
  }
}
