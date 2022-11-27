#include <bits/stdc++.h>

using namespace std;
typedef long long ll;

const int mxn = 500 + 6;
const int mxm = 1e5 + 6;
const int inf = 0x3f3f3f3f;
int n, m, k;
struct Edge {
  int u, v, t, c, next;
} E[mxm];
int head[mxn];
int d[mxn];
deque<int> Q;
int narrow[mxn]; // s u

void addEdge(int u, int v, int t, int c) {
  E[k].u = u, E[k].v = v, E[k].t = t, E[k].c = c;
  E[k].next = head[u];
  head[u] = k++;
}

void spfa(int s) {
  d[s] = 0;
  narrow[s] = 0;
  Q.push_back(s);
  while (!Q.empty()) {
    int u = Q.front();
    Q.pop_front();
    for (int i = head[u]; i != -1; i = E[i].next) {
      int v = E[i].v, t = E[i].t, c = E[i].c;
      if (t) {
        if (d[v] >
            d[u] - narrow[u] * narrow[u] + (narrow[u] + c) * (narrow[u] + c)) {
          d[v] =
              d[u] - narrow[u] * narrow[u] + (narrow[u] + c) * (narrow[u] + c);
          narrow[v] = narrow[u] + c;
          Q.push_back(v);
        }
      } else {
        if (d[v] > d[u] + c) {
          d[v] = d[u] + c;
          narrow[v] = 0;
          Q.push_back(v);
        }
      }
      // printf("%d %d %d %d %d %d\n", u,v,t,c,d[u],d[v]);
    }
  }
}

int main() {
  ios::sync_with_stdio(false);
#ifdef LOCAL
  freopen("in.txt", "r", stdin);
#endif
  scanf("%d%d", &n, &m);
  memset(head, -1, sizeof(head));
  int u, v, t, c;
  for (int i = 0; i < m; ++i) {
    scanf("%d%d%d%d", &t, &u, &v, &c);
    addEdge(u, v, t, c);
    addEdge(v, u, t, c);
  }
  memset(d, inf, sizeof(d));
  spfa(1);
  printf("%d\n", d[n]);
  return 0;
}