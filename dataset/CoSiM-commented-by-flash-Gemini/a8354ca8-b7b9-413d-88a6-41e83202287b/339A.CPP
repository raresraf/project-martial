#include <bits/stdc++.h>
using namespace std;
int main() {
  string str;
  int i = 0, n = 0, t = 0, e = 0;
  cin >> str;
  for (int j = 0; j < str.size(); j++) {
    if (str[j] == 'n')
      n++;
    else if (str[j] == 't')
      t++;
    else if (str[j] == 'e')
      e++;
    else if (str[j] == 'i')
      i++;
    else {
    }
  }
  n--;
  e /= 3;
  n /= 2;
  cout << min(n, min(e, min(i, t)));
  return 0;
}
