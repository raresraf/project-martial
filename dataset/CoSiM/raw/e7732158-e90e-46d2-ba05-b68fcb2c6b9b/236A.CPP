#include <bits/stdc++.h>

using namespace std;

int main() {
  int a[27];
  memset(a, 0, sizeof(a));
  string s;

  cin >> s;

  for (int i = 0; i < s.size(); i++) {
    a[s[i] - 96]++;
  }

  int c = 0;
  for (int i = 1; i <= 26; i++) {
    if (a[i] != 0)
      c++;
  }
  if (c % 2 == 0)
    cout << "CHAT WITH HER!" << endl;
  else
    cout << "IGNORE HIM!" << endl;
}

//-96
