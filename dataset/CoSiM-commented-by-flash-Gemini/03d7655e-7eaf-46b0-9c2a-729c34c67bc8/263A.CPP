// https://codeforces.com/contest/263/problem/A
#include <bits/stdc++.h>
using namespace std;
#define in cin >>
#define out cout <<
#define FORI(i, k, l) for (int i(k); i < l; i++)
#define FORJ(i, k, l) for (int j(k); j < l; j++)
#define Vi vector<int>
#define Msi map<string, int>
#define Mii map<int, int>

int main(void) {
  int arr[5][5];
  int x = 0, y = 0;

  FORI(i, 0, 5) {
    FORJ(j, 0, 5) {
      in(arr[i][j]);
      if (arr[i][j] == 1) {
        x = j + 1;
        y = i + 1;
      }
    }
  }
  x = abs(x - 3);
  y = abs(y - 3);
  out(x + y);
}