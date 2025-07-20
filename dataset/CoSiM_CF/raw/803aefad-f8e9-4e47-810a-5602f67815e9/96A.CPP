
// Problem Link https://codeforces.com/problemset/problem/96/A

#include <iostream>

using namespace std;

int main() {
  string str;
  cin >> str;

  int count = 0;
  char l = '2';
  for (int i = 0; i < str.length(); i++) {
    if ((str[i] == '0' && l == '1') || (str[i] == '1' && l == '0')) {
      count = 0;
    }
    count++;
    l = str[i];
    if (count == 7) {
      break;
    }
  }

  if (count == 7) {
    cout << "YES\n";
  } else {
    cout << "NO\n";
  }
}