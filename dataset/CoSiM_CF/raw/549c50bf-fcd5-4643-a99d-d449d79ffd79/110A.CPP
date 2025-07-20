#include <bits/stdc++.h>
using namespace std;
int main() {
  string s;
  cin >> s;
  int cnt = 0;
  for (int i = 0; i < s.length(); i++) {
    if (s[i] == '7' || s[i] == '4')
      cnt++;
  }
  if (cnt == 4 || cnt == 7)
    cout << "YES" << endl;
  else
    cout << "NO";
  return 0;
}
