#include <bits/stdc++.h>
using namespace std;
int main() {
  string str;
  int h = 0, e = 0, l = 0, ll = 0, i, flag = 0;
  cin >> str;
  for (i = 0; i <= str.size(); i++) {
    if (str[i] == 'h')
      h = 1;
    else if (str[i] == 'e' && h == 1)
      e = 1;
    else if (str[i] == 'l' && h == 1 && e == 1)
      l += 1;
    else if (str[i] == 'o' && h == 1 && e == 1 && l > 1)
      flag = 1;
  }
  if (flag == 1)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
}
