// https://codeforces.com/contest/231/problem/A
#include <bits/stdc++.h>
using namespace std;
#define in cin >>
#define out cout <<
#define FOR(i, k, l) for (int i(k); i < l; i++)
#define Vi vector<int>
#define Msi map<string, int>
#define Mii map<int, int>

int main(void) {
  int n;
  in(n);
  int a, b, c, res = 0;

  FOR(i, 0, n) {
    int s = 0;
    in(a);
    in(b);
    in(c);
    if (a == 1)
      s++;
    if (b == 1)
      s++;
    if (c == 1)
      s++;

    if (s >= 2)
      res++;
  }

  out(res);
}