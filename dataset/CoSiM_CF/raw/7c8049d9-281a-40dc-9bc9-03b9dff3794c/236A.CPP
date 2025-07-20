#include <bits/stdc++.h>

using namespace std;

int main() {
  ios::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);
  string str, s2;
  cin >> str;
  sort(str.begin(), str.end());
  auto res = unique(str.begin(), str.end());
  s2 = string(str.begin(), res);
  int n = s2.length();

  if (n % 2 == 0) {
    cout << "CHAT WITH HER!";
  } else {
    cout << "IGNORE HIM!";
  }
  return 0;
}
