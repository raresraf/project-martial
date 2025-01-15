#include <bits/stdc++.h>
using namespace std;

int main() {
  string str;
  cin >> str;

  bool isCap = (int)str[0] < 97;

  if (isCap) {
    cout << str << endl;
  } else {
    cout << (char)((int)str[0] - 97 + 65);
    cout << str.substr(1, str.length()) << endl;
  }

  return 0;
}
