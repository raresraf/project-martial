
#include <bits/stdc++.h>

using namespace std;

int n, a[200];
int sum;

int main() {
  scanf("%d", &n);
  for (int i = 0; i < n; i++)
    scanf("%d", &a[i]), sum += a[i];
  sort(a, a + n);
  int ans = 0, my = 0, i = n - 1;
  while (my <= sum - my) {
    my += a[i--];
    ans++;
  }
  printf("%d", ans);
}
