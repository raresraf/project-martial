#include <algorithm>
#include <bits/stdc++.h>
#include <iostream>
#include <string>
using namespace std;
int main() {
  string str, str2;
  cin >> str;
  for (int i = 0; i < str.size(); i++) {
    if (str[i] != '+') {
      str2.push_back(str[i]);
    }
  }
  sort(str2.begin(), str2.end());
  cout << str2[0];
  for (int i = 1; i < str2.size(); i++) {
    cout << '+' << str2[i];
  }
  cout << '\n' << endl;

  // cout<<str<<endl;
}
