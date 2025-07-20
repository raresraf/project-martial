// http://codeforces.com/problemset/problem/110/A
#include <bits/stdc++.h>
using namespace std;

int islucky(int a) {
  // returns 1 if a only has 4s and 7s
  if (a <= 1)
    return 0;
  while (a >= 1) {
    if (a % 10 != 4 && a % 10 != 7)
      return 0;
    a = a / 10;
  }
  return 1;
}

int main() {
  string num;
  cin >> num;
  int luckydigs = 0;
  for (int i = 0; i < num.size(); i++) {
    if (num[i] == '4' || num[i] == '7')
      luckydigs++;
  }
  if (islucky(luckydigs)) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }
  return 0;
}
