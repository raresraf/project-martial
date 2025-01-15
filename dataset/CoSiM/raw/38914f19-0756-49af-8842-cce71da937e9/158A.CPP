#include <cstdio>
#include <iostream>

using namespace std;

int main() {
  int i, n, k, score, threshold = -1;
  scanf("%d %d", &n, &k);

  for (i = 0; i < n; ++i) {
    scanf("%d", &score);
    if (score == 0)
      break;
    if (i <= k - 1) {
      threshold = score;
    }
    if (threshold != -1) {
      if (score < threshold) {
        break;
      }
    }
  }
  printf("%d\n", i);
}