#include <bits/stdc++.h>
using namespace std;
int main() {
  int a[100];
  string s;
  cin >> s;
  a[0] = 2;
  a[1] = 7;
  a[2] = 2;
  a[3] = 3;
  a[4] = 3;
  a[5] = 4;
  a[6] = 2;
  a[7] = 5;
  a[8] = 1;
  a[9] = 2;
  int first, second;
  first = s[0] - '0';
  second = s[1] - '0';
  int sum = a[first] * a[second];
  cout << sum << endl;
  return 0;
}
