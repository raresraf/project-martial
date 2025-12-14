#include <bits/stdc++.h>
using namespace std;
int main() {
  int step = 0, sumt = 0, n;
  int sumMe = 0, i;
  scanf("%d", &n);
  int A[n];
  for (i = 0; i < n; scanf(" %d", (A + i)), sumt += A[i++])
    ;
  // for (i = 0; i < n; printf("%d ", A[i++]));
  // printf("%d %d\n", sumt, sumMe);
  sort(A, A + n);
  // for (i = 0; i < n; printf("%d ", A[i++]));
  // printf("\n");
  i = n - 1;
  while (i >= 0 && sumt >= sumMe) {
    sumt -= A[i];
    sumMe += A[i];
    ++step;
    i--;
  }
  printf("%d\n", step);
  return 0;
}
