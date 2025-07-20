// 236A.cpp
// Boy or Girl
// Author: Ikki
// May 31, 2015

#include <iostream>
#include <string>
using namespace std;

bool f[26];

int main() {
  string str;
  cin >> str;
  for (int i = 0; i < str.size(); i++)
    f[str[i] - 'a'] = true;
  int cnt = 0;
  for (int i = 0; i < 26; i++)
    if (f[i])
      cnt++;
  if (cnt % 2 == 0)
    cout << "CHAT WITH HER!" << endl;
  else
    cout << "IGNORE HIM!" << endl;
  return 0;
}