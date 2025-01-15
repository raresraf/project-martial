#include "bits/stdc++.h"

using namespace std;

typedef std::vector<int> vi;
typedef vector<vector<int>> vvi;

int main(int argc, char const *argv[]) {
  vvi v(5, vi(5));

  int a, b;

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> v[i][j];
      if (v[i][j] == 1) {
        a = i;
        b = j;
      }
    }
  }

  a = (abs(a - 2) + abs(b - 2));

  std::cout << a;

  return 0;
}
