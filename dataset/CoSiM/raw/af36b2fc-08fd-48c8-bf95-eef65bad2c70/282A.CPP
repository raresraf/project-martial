// https://codeforces.com/problemset/problem/282/A
// 282A  Bit++
#include <bits/stdc++.h>
using namespace std;
int main() {
  int n;
  cin >> n;
  string str;
  int i = 0, j = 0;
  while (n--) {
    cin >> str;
    if (str[1] == '+') {
      i++;
    } else
      j--;
  }
  cout << i + j;
}
