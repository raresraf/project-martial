#include <algorithm>
#include <stdio.h>

using namespace std;

int main() {
  int n, t;
  scanf("%d %d", &n, &t);
  char queue[n];
  scanf("%s", &queue);
  while (t--) {
    for (int i = 0; i < n - 1; i++) {
      if (queue[i] == 'B' && queue[i + 1] == 'G')
        swap(queue[i], queue[i + 1]), i++;
    }
  }
  printf("%s\n", queue);
  return 0;
}