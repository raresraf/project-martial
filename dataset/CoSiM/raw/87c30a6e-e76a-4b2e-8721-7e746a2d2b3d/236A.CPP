// 236A - Boy or Girl

#include <bits/stdc++.h>
using namespace std;

int main() {
  string s;
  cin >> s;

  int hist[26] = {0};
  for (int i = 0; i < s.size(); i++)
    hist[s[i] - 97] = 1;
  int cnt = 0;
  for (int i = 0; i < 26; i++) {
    if (hist[i] > 0)
      cnt++;
  }

  if (cnt % 2 == 1)
    cout << "IGNORE HIM!";
  else
    cout << "CHAT WITH HER!";
  return 0;
}