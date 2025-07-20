// https://codeforces.com/problemset/problem/281/A
// 281A. Word Capitalization

#include <bits/stdc++.h>
using namespace std;

int main() {
  string str;
  cin >> str;
  str[0] = toupper(str[0]);
  cout << str;
}
