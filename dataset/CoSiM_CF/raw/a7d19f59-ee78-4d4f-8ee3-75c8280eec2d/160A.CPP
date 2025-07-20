#include <malloc.h>
#include <stdio.h>

void sort(int list[], int size) {
  int i, j, minIdx, tmp;

  for (i = 0; i < size - 1; i++) {
    minIdx = i;
    for (j = i + 1; j < size; j++) {
      if (list[minIdx] > list[j]) {
        minIdx = j;
      }
    }

    if (minIdx != i) {
      tmp = list[i];
      list[i] = list[minIdx];
      list[minIdx] = tmp;
    }
  }
}

int main(void) {

  int n;
  int m_coin;
  int t_coin;
  int *a;

  scanf("%d", &n);

  a = (int *)malloc(n * sizeof(int)); //

  for (int i = 0; i < n; i++) {
    scanf("%d ", &a[i]);
  }

  sort(a, n);

  for (int mine = 1; mine <= n; mine++) {
    m_coin = 0;
    t_coin = 0;
    for (int x = 0; x < n - mine; x++) {
      t_coin += a[x];
    }
    for (int y = n - mine; y < n; y++) {
      m_coin += a[y];
    }
    if (m_coin > t_coin) {
      printf("%d", mine);
      break;
    }
  }
  return 0;
}
