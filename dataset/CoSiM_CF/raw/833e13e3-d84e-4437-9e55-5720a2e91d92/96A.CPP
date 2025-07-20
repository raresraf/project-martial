#include <bits/stdc++.h>

using namespace std;

int main() {
  int a = 0, b = 0, t = 0;
  string str;
  cin >> str;
  for (int i = 0; i < str.length(); i++) {
    if (str[i] == '1') {
      a++;
      b = 0;
    } else {
      b++;
      a = 0;
    }
    if (a == 7 || b == 7) {
      t = 1;
      break;
    }
  }
  if (t == 1)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
  return 0;
}
