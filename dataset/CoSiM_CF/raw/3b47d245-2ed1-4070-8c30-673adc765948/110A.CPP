#include <bits/stdc++.h>

using namespace std;

int main() {
  string s;
  int count = 0;
  cin >> s;

  int l = s.length();

  for (int i = 0; i < l; i++) {
    if (s[i] == '4' || s[i] == '7')
      count++;
  }

  if (count == 4 || count == 7)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
}
