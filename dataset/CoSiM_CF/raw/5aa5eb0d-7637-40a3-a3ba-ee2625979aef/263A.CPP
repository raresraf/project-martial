#include <iostream>

using namespace std;

#define long long ll;

int matrix[5][5];
int r, m;

void solve() {
  int mvs = abs(2 - r) + abs(2 - m);
  cout << mvs;
}

int main() {
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> matrix[i][j];
      if (matrix[i][j]) {
        r = i;
        m = j;
      }
    }
  }
  solve();
}