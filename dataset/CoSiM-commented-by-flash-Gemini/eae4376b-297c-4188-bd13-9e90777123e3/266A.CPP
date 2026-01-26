// 266A.cpp
// Stones on the Table
// Author: Ikki
// May 31, 2015

#include <iostream>
#include <string>
using namespace std;

int main() {
  int n;
  cin >> n;
  string str;
  cin >> str;
  int ans = 0;
  char prev = str[0];
  for (int i = 1; i < str.size(); i++) {
    if (str[i] == prev)
      ans++;
    else
      prev = str[i];
  }
  cout << ans << endl;
  return 0;
}