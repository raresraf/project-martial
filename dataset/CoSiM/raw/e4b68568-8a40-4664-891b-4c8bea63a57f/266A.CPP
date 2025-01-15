//
// Created by Taro Young on 2020/9/7.
//

#include <iostream>
#include <string>

using namespace std;

int main() {
  int n;
  int num = 0;
  cin >> n;
  string s;
  cin >> s;
  for (int i = 1; i < n; i++) {
    if (s[i] == s[i - 1])
      num++;
  }
  cout << num;
  return 0;
}