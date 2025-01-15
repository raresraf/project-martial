//
// Created by Taro Young on 2020/9/7.
//
#include <iostream>
#include <string>

using namespace std;

int a[30];

int main() {
  string s;
  cin >> s;
  int num = 0;
  for (char ch : s)
    a[ch - 'a']++;
  for (int i = 0; i < 26; i++)
    if (a[i] != 0)
      num++;
  if (num % 2 == 0)
    cout << "CHAT WITH HER!";
  else
    cout << "IGNORE HIM!";
  return 0;
}
