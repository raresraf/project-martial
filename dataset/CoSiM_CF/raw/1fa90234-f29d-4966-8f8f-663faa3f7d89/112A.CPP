// https://codeforces.com/contest/112/problem/A
#include <bits/stdc++.h>
using namespace std;
#define in cin >>
#define out cout <<
#define FOR(i, k, l) for (int i(k); i < l; i++)
#define Vi vector<int>
#define Msi map<string, int>
#define Mii map<int, int>

int main(void) {
  string a, b;
  in(a);
  in(b);
  transform(a.begin(), a.end(), a.begin(), ::tolower);
  transform(b.begin(), b.end(), b.begin(), ::tolower);
  out(strcmp(a.c_str(), b.c_str()));
}