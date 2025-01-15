#include <bits/stdc++.h>
using namespace std;

int main() {
  bool flag = false;
  int j = 0;
  string s;
  char ch[] = {'h', 'e', 'l', 'l', 'o'};
  cin >> s;
  for (int i = 0; i < s.size(); i++) {
    if (s[i] == ch[j])
      j++;
    if (j > 4) {
      flag = true;
      break;
    }
  }
  if (flag)
    cout << "YES\n";
  else
    cout << "NO\n";
  return 0;
}
