#include <iostream>
#include <vector>

/*
 * Contest: VK Cup 2012 Q1 (Task 158-A)
 * URL: http://codeforces.ru/contest/158/problem/A
 */

int main(int argc, char **argv) {
  int n, k, count = 0;
  std::cin >> n >> k;

  std::vector<int> scores(n);

  for (unsigned i = 0; i < n; ++i) {
    std::cin >> scores[i];

    if (scores[i] > 0 && (i < k || (i >= k && scores[i] == scores[i - 1]))) {
      count++;
    } else {
      break;
    }
  }

  std::cout << count << std::endl;
  return 0;
}
