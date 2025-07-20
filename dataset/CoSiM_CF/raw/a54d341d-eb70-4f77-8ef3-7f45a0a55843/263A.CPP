/*
 *	User: Isanchez_Aguilar
 *	Problem: CodeForces 263A - Beautiful Matrix
 */
#include <bits/stdc++.h>

using namespace std;

int main(void) {
  ios_base::sync_with_stdio(0);
  cin.tie(0);

  int oneI = 0;
  int oneJ = 0;
  int mat[5][5];

  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      cin >> mat[i][j];

      if (mat[i][j] == 1)
        oneI = i, oneJ = j;
    }
  }

  cout << abs(oneI - 2) + abs(oneJ - 2) << "\n";

  return 0;
}