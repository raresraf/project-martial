// 110A.cpp
// A. Nearly Lucky Number
// Author: Ikki
// Jun 10, 2015

#include <iostream>
#include <string>
using namespace std;

int main() {
  string str;
  cin >> str;
  int cnt = 0;
  for (int i = 0; i < (int)str.size(); i++) {
    if (str[i] == '4' || str[i] == '7')
      cnt++;
  }
  if (cnt != 4 && cnt != 7)
    cout << "NO" << endl;
  else
    cout << "YES" << endl;
  return 0;
}