//
// Created by Taro Young on 2020/9/7.
//
#include <iostream>
#include <string>

using namespace std;

int main() {
  string s;
  cin >> s;
  int num = 1;
  bool ans = false;
  for (int i = 1; i < s.length(); i++) {
    if (s[i] == s[i - 1])
      num++;
    else
      num = 1;
    if (num == 7) {
      ans = true;
      break;
    }
  }
  if (ans)
    cout << "YES";
  else
    cout << "NO";
  return 0;
}
