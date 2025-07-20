#include <bits/stdc++.h>
using namespace std;
int main() {
  int len, zero = 0, one = 0, flag = 0;
  string s;
  cin >> s;
  len = s.length();
  for (int i = 0; i < len; i++) {
    if (s[i] == '0')
      zero++;
    else
      zero = 0;
    if (s[i] == '1')
      one++;
    else
      one = 0;
    if (zero == 7 || one == 7) {
      flag = 1;
      break;
    }
  }
  if (flag == 1)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
}
