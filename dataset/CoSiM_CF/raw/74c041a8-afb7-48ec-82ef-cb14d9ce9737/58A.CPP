// 58A.cpp
// A. Chat room
// Author: Ikki
// Jun 10, 2015

#include <iostream>
#include <string>
using namespace std;

int main() {
  string str;
  cin >> str;
  string hello = "hello";
  int j = 0;
  for (int i = 0; i < (int)str.size() && j < 5; i++)
    if (str[i] == hello[j])
      j++;
  cout << (j == 5 ? "YES" : "NO") << endl;
  return 0;
}