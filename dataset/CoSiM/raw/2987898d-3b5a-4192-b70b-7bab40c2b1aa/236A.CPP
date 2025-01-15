#include <bits/stdc++.h>
using namespace std;

int main() {
  ios::sync_with_stdio(false);
  cin.tie(0);

  string s;
  cin >> s;

  sort(s.begin(), s.end());
  s.erase(unique(s.begin(), s.end()), s.end());

  if ((s.length()) % 2 == 0)
    cout << "CHAT WITH HER!\n";

  else
    cout << "IGNORE HIM!\n";
}
