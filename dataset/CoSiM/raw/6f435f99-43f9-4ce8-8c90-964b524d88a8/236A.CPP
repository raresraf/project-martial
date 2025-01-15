#include <bits/stdc++.h>
#include <iostream>
using namespace std;
int main() {
  string uname, temp;
  cin >> uname;
  int flag, k = 0;
  for (int i = 0; i < uname.length(); i++) {
    flag = 0;
    for (int j = 0; j < temp.length(); j++)
      if (uname[i] == temp[j])
        flag++;
    if (flag == 0)
      temp += uname[i];
  }

  if (temp.length() % 2)
    cout << "IGNORE HIM!";
  else
    cout << "CHAT WITH HER!";
}
