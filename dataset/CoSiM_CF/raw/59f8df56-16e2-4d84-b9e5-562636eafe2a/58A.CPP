#include <bits/stdc++.h>

using namespace std;

int main() {
  string s;
  cin >> s;

  string h = "hello";
  int idx = 0;

  for (int i = 0; i < s.size(); i++) {
    if (idx == 5)
      break;
    if (s[i] == h[idx])
      idx++;
  }

  if (idx == 5)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}
