// https://codeforces.com/problemset/problem/110/A
// 110A. Nearly Lucky Number

#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
  ll n;
  cin >> n;
  int flag = 0;

  if (n < 1) {
    flag == 1;
  }

  string num = to_string(n);
  for (int i = 0; i < num.size(); i++) {
    if (num[i] == '7' || num[i] == '4') {
      flag = 1;
    }
  }
  if (flag == 1) {
    cout << "YES";
  } else {
    cout << "NO";
  }
  return 0;
}
