#include <cstdio>

const int N = 3;
int force[N];

int main() {
  int n;
  scanf("%d", &n);

  for (int i = 0; i < n * N; ++i) {
    int m;
    scanf("%d", &m);
    force[i % 3] += m;
  }

  bool ok = true;

  for (int i = 0; i < N; ++i) {
    if (force[i]) {
      ok = false;
      break;
    }
  }

  printf(ok ? "YES\n" : "NO\n");
  return 0;
}
