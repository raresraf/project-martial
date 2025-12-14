// 96A.cpp
// A. Football
// Author: Ikki
// May 31, 2015

#include <iostream>
#include <string>
using namespace std;

int main() {
  string str;
  cin >> str;
  int cnt = 1;
  for (int i = 1; i < str.size(); i++) {
    if (str[i] == str[i - 1]) {
      cnt++;
      if (cnt >= 7) {
        cout << "YES" << endl;
        return 0;
      }
    } else
      cnt = 1;
  }
  cout << "NO" << endl;
  return 0;
}