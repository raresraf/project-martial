#include <bits/stdc++.h>
#include <iostream>
#include <string>
using namespace std;

int main() {
  int n;
  cin >> n;
  string str;
  for (int i = 0; i <= n; i++) {
    getline(cin, str);

    if (str.size() > 10) {
      cout << str[0] << str.size() - 2 << str[str.size() - 1] << endl;

    } else {
      cout << str << endl;
    }
  }
}
