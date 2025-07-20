// http://codeforces.com/problemset/problem/282/A

#include <iostream>
using namespace std;

int main() {
  int n, res = 0;
  string s;
  cin >> n;
  while (n--) {
    cin >> s;
    if (s[0] == '+' || s[2] == '+')
      res++;
    else
      res--;
  }
  cout << res;

  return 0;
}
