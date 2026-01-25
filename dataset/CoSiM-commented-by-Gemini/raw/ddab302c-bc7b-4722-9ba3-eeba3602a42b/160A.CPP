#include <iostream>
#include <stdlib.h>

int compare(const void *a, const void *b) { return (*(int *)a - *(int *)b); }

int main() {
  int flag = 0, i, n, value[105], sum = 0, num = 0;
  std::cin >> n;
  for (i = 0; i < n; i++) {
    std::cin >> value[i];
    sum = sum + value[i];
  }
  if (sum % 2 == 0) {
    sum = sum / 2;
    flag = 1;
  } else
    sum = (sum + 1) / 2;
  qsort(value, n, sizeof(int), compare);
  for (i = n - 1; i >= 0; i--) {
    if (sum >= value[i]) {
      num++;
      sum = sum - value[i];
    } else if (sum < value[i]) {
      if (sum != 0)
        num++;
      break;
    }
  }
  if (flag == 1 && sum == 0)
    num++;
  std::cout << num << '\n';
  return 0;
}
