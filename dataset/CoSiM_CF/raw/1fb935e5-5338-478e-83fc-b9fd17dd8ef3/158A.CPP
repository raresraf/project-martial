#include <bits/stdc++.h>

using namespace std;

int num[55];

int main() {

  int n, k;

  scanf("%d%d", &n, &k);

  for (int i = 1; i <= n; ++i)
    scanf("%d", &num[i]);

  int passing = num[k];

  int count = 0;

  if (passing == 0) {

    int i;

    for (i = k; i > 0; --i)
      if (num[i] > 0)
        break;

    count = i;
    ;
  } else {

    count = k;

    int i;

    for (i = k + 1; i <= n; ++i) {
      if (num[i] < passing)
        break;
      else
        count++;
    }
  }

  printf("%d\n", count);
}