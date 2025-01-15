#include <iostream>

int main() {
  int n, k;
  std::cin >> n >> k;
  int *scores = new int[n];
  for (int i = 0; i < n; i++) {
    std::cin >> scores[i];
  }
  int index = 0;
  int count = 0;
  while (index <= k && scores[index] != 0) {
    if (index == k - 1) {
      int val = scores[index];
      count++;
      for (; k < n; k++) {
        if (scores[k] == val)
          count++;
        else
          break;
      }
      break;
    } else
      count++;
    index++;
  }
  std::cout << count << std::endl;
  delete scores;
  return 0;
}
