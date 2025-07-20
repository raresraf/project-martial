#include <bits/stdc++.h>

using namespace std;

int N = 5;

int main() {

  int x, y;

  int mat[N + 1][N + 1];

  for (int i = 1; i <= N; ++i)
    for (int j = 1; j <= N; ++j) {

      scanf("%d", &mat[i][j]);

      if (mat[i][j] == 1)
        x = i, y = j;
    }

  int count = abs(x - 3) + abs(y - 3);

  cout << count << endl;
}