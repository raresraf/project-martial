
#include <bits/stdc++.h>

using namespace std;

int cnt, a[100], n, kth, k;

int main() {
  scanf("%d%d", &n, &k);
  for (int i = 0; i < n; i++)
    scanf("%d", &a[i]);
  kth = a[k - 1];
  for (int i = 0; i < n; i++)
    cnt += (a[i] >= kth && a[i] > 0);
  printf("%d", cnt);
}
