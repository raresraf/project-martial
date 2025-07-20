#include <bits/stdc++.h>

using namespace std;

int main() {
  string str1;
  cin >> str1;
  string str2;
  cin >> str2;
  transform(str1.begin(), str1.end(), str1.begin(), ::tolower);
  transform(str2.begin(), str2.end(), str2.begin(), ::tolower);
  int len = 0;
  while (len++ < str1.size()) {
    if (str1[len] > str2[len]) {
      cout << 1;
      return 0;
    } else if (str1[len] < str2[len]) {
      cout << -1;
      return 0;
    } else
      continue;
  }
  cout << 0;
  return 0;
}