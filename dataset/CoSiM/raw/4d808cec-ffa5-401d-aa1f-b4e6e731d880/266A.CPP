#include <bits/stdc++.h>
#include <iostream>
using namespace std;
int main() {
  string str;
  int n, j;
  cin >> n;
  cin >> str;
  int min = 0;
  int i = 0;
  for (; i < str.length();) {
    if (str[i] == str[i + 1]) {
      str.erase(str.begin() + i);
      min++;
    } else
      i++;
  }
  cout << min;
  return 0;
}
