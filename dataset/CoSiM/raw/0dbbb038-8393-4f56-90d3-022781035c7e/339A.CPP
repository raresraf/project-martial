#include <bits/stdc++.h>
using namespace std;

int main() {
  string a;
  bool ck = false;
  cin >> a;
  int len = a.length(), num = len / 2 + 1, num1 = 0, num2 = 0, num3 = 0;
  for (int i = 0; i < len; i++) {
    if (a[i] == '1') {
      num1++;
    }
    if (a[i] == '2') {
      num2++;
    }
    if (a[i] == '3') {
      num3++;
    }
  }
  for (int i = 0; i < num1; i++) {
    if (ck == false) {
      ck = true;
    } else {
      cout << '+';
    }
    cout << 1;
  }
  for (int i = 0; i < num2; i++) {
    if (ck == false) {
      ck = true;
    } else {
      cout << '+';
    }
    cout << 2;
  }
  for (int i = 0; i < num3; i++) {
    if (ck == false) {
      ck = true;
    } else {
      cout << '+';
    }
    cout << 3;
  }
}
