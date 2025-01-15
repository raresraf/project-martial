// By archit-dev, contest: Codeforces Beta Round #87 (Div. 2 Only), problem: (A)
// Tram, Accepted, #, Copy

#include <algorithm>
#include <bits/stdc++.h>
#include <cmath>
#include <string>
#define ll long long int
#define ull unsigned long long int
using namespace std;

int main() {
#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif

  int n;
  cin >> n;
  int capa = 0, curr = 0;
  int a, b;
  cin >> a >> b;
  curr = b;
  capa = b;
  for (int i = 1; i < n; i++) {
    cin >> a >> b;
    curr = curr + b - a;
    capa = max(capa, curr);
  }
  cout << capa << "\n";
  return 0;
}
