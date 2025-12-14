#include <bits/stdc++.h>
using namespace std;
int main() {
  string s;
  int x = 0;
  cin >> s;
  sort(s.begin(), s.end());
  for (int i = 0; i < s.size(); i++) {
    if (s[i] != s[i + 1]) {
      x++;
    }
  }
  if (x % 2 == 0)
    cout << "CHAT WITH HER!" << endl;
  else
    cout << "IGNORE HIM!" << endl;
}
