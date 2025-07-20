#include <bits/stdc++.h>

using namespace std;

int main() {
  string exp;

  getline(cin, exp);
  int o = (exp.length() + 1) / 2;
  int num[o];
  for (int i = 0, j = 0; i < exp.length(); i += 2, j++) {
    num[j] = int(exp[i]) - 48;
  }
  sort(num, num + o);
  for (int i = 0; i < (o - 1); i++) {
    cout << num[i] << "+";
  }
  cout << num[o - 1];
}