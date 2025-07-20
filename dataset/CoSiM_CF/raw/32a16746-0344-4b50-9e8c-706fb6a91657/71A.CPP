// 71A.cpp
// A. Way Too Long Words
// Author: derekhh
// May 31, 2015

#include <iostream>
#include <string>
using namespace std;

int main() {
  int n;
  cin >> n;
  for (int i = 0; i < n; i++) {
    string str;
    cin >> str;
    if (str.size() <= 10)
      cout << str << endl;
    else
      cout << str[0] << str.size() - 2 << str[str.size() - 1] << endl;
  }
  return 0;
}