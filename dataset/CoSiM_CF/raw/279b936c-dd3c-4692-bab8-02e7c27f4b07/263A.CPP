#include <bits/stdc++.h>
#define le 6
using namespace std;
int n[le][le];
bool vis[le][le];
int dis[le][le];
int fx[] = {1, -1, 0, 0};
int fy[] = {0, 0, 1, -1};
int bfs(int a, int b, int r, int c) {
  for (int i = 0; i < le; i++)
    for (int j = 0; j < le; vis[i][j] = false, dis[i][j] = 0, j++)
      ;
  vis[a][b] = true;
  dis[a][b] = 0;
  queue<pair<int, int>> q;
  q.push(make_pair(a, b));
  while (!q.empty()) {
    pair<int, int> p = q.front();
    q.pop();
    for (int i = 0; i < 4; i++) {
      int py = p.first + fy[i], px = p.second + fx[i];
      if (py >= 0 && py < r && px >= 0 && px < c && vis[py][px] == false) {
        vis[py][px] = true;
        dis[py][px] = dis[p.first][p.second] + 1;
        q.push(make_pair(py, px));
      }
    }
  }
  return dis[2][2];
}
int main() {
  // freopen("input.txt", "r", stdin);
  int r = 5, c = 5, a, x, y;
  for (int i = 0; i < r; i++) {
    for (int j = 0; j < c; j++) {
      scanf("%d", &a);
      n[i][j] = a;
      if (a) {
        y = i;
        x = j;
      }
    }
  }
  printf("%d\n", bfs(y, x, r, c));
  return 0;
}
