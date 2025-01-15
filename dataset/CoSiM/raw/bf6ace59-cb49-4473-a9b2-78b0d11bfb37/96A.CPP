#include <bits/stdc++.h>
using namespace std;
int main() {
  string str;
  cin >> str;
  int a = 1, b = 0;
  for (int i = 0; i < str.size(); i++) {
    if (str[i] == str[i + 1]) {
      a++;
      if (a == 7) {
        b = 1;
        break;
      }
    } else
      a = 1;
  }

  if (b == 1)
    cout << "YES";
  else
    cout << "NO";
  return 0;
}
