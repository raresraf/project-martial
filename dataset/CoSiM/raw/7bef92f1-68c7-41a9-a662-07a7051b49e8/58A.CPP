#include <bits/stdc++.h>
using namespace std;
int main() {
  string s;
  cin >> s;
  int helo[4] = {0}, flag = 0;
  for (int i = 0; i < s.size(); i++) {
    if (s[i] == 'h') {
      helo[0]++;
    } else if (s[i] == 'e' && helo[0] >= 1) {
      helo[1]++;
    } else if (s[i] == 'l' && helo[1] >= 1) {
      helo[2]++;
    } else if (s[i] == 'o' && helo[2] >= 2) {
      flag = 1;
      break;
    }
  }
  if (flag)
    cout << "YES";
  else
    cout << "NO";

  return 0;
}
