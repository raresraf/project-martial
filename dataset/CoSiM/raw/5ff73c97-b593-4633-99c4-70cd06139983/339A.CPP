//
// Created by Taro Young on 2020/9/7.
//

#include <iostream>
#include <string>

using namespace std;

int main() {
  string s;
  cin >> s;
  int num1 = 0;
  int num2 = 0;
  int num3 = 0;
  for (char ch : s) {
    if (ch == '1')
      num1++;
    if (ch == '2')
      num2++;
    if (ch == '3')
      num3++;
  }
  int first = num1 != 0 ? 1 : num2 != 0 ? 2 : 3;
  if (first == 1)
    num1--;
  if (first == 2)
    num2--;
  if (first == 3)
    num3--;
  cout << first;
  for (int i = 0; i < num1; i++)
    cout << '+' << 1;
  for (int i = 0; i < num2; i++)
    cout << '+' << 2;
  for (int i = 0; i < num3; i++)
    cout << '+' << 3;
  return 0;
}