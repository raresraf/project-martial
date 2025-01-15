#include <bits/stdc++.h>
using namespace std;
int main() {
  string name;
  cin >> name;

  int cnt = 0, len = name.length();
  sort(name.begin(), name.end());
  for (int i = 0; i < len; i++) {
    if (name[i] != name[i + 1])
      cnt++;
  }

  if (cnt % 2 != 0)
    cout << "IGNORE HIM!";
  else
    cout << "CHAT WITH HER!";
}
