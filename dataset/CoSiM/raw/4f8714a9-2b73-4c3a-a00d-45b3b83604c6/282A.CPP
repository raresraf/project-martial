#include <bits/stdc++.h>
#include <iostream>
using namespace std;
int main() {
  int n;
  cin >> n;
  string s;
  int res = 0;
  while (n--) {
    cin >> s;
    if (s[1] == '+')
      res++;
    else
      res--;
  }
  cout << res;
}
